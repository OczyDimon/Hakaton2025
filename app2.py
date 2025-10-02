import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import io
import base64
import faiss
import time
from tqdm import tqdm
import gc
import pandas as pd


class EfficientCarCameraDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, os.path.basename(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Возвращаем черное изображение в случае ошибки
            return torch.zeros(3, 224, 224), os.path.basename(img_path)


class OptimizedResNetEmbeddingExtractor(nn.Module):
    def __init__(self, embedding_dim=512):
        super(OptimizedResNetEmbeddingExtractor, self).__init__()

        # Используем ResNet50 как баланс между качеством и скоростью
        self.backbone = models.resnet50(pretrained=True)

        # Убираем последний слой и используем предпоследние features
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-2])

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Проекция до 512 измерений
        self.projection = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        pooled = self.global_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        embeddings = self.projection(flattened)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

def get_image_paths(image_dir, max_images=500000):
    """Рекурсивно получает пути ко всем изображениям"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
                if len(image_paths) >= max_images:
                    return image_paths[:max_images]

    return image_paths

def create_embeddings_colab(image_dir, output_filename="moscow_embeddings",
                          batch_size=32, embedding_dim=512, max_images=500000):
    """
    Оптимизированная функция для создания эмбеддингов
    """
    # Проверяем доступность GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Трансформы с аугментацией для тестирования
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Получаем пути к изображениям
    print("Scanning for images...")
    image_paths = get_image_paths(image_dir, max_images)
    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        raise ValueError("No images found in the specified directory")

    # Создаем датасет и даталоадер
    dataset = EfficientCarCameraDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                          shuffle=False, num_workers=2,
                          pin_memory=True)

    # Инициализируем модель
    model = OptimizedResNetEmbeddingExtractor(embedding_dim=embedding_dim)
    model = model.to(device)
    model.eval()

    # Для отслеживания прогресса и сохранения результатов
    all_embeddings = []
    all_filenames = []

    print("Starting embedding extraction...")

    with torch.no_grad():
        for batch_idx, (images, filenames) in enumerate(tqdm(dataloader)):
            images = images.to(device, non_blocking=True)

            # Получаем эмбеддинги
            embeddings = model(images)

            # Сохраняем на CPU
            all_embeddings.append(embeddings.cpu().numpy())
            all_filenames.extend(filenames)

            # Периодически чистим память и сохраняем прогресс
            if batch_idx % 50 == 0 and batch_idx > 0:
                # Сохраняем промежуточные результаты
                temp_embeddings = np.vstack(all_embeddings)
                temp_results = {
                    'embeddings': temp_embeddings,
                    'filenames': all_filenames
                }
                temp_path = f"files/{output_filename}_temp.npy"
                np.save(temp_path, temp_results)
                print(f"Saved checkpoint at batch {batch_idx}")

                # Чистим память
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

    # Финальное сохранение
    all_embeddings = np.vstack(all_embeddings)

    results = {
        'embeddings': all_embeddings,
        'filenames': all_filenames,
        'embedding_dim': embedding_dim,
        'total_images': len(all_filenames)
    }

    # Сохраняем в разных форматах
    output_path_npy = f"/content/{output_filename}.npy"
    output_path_csv = f"/content/{output_filename}_metadata.csv"

    np.save(output_path_npy, results)

    # Сохраняем метаданные в CSV
    metadata_df = pd.DataFrame({
        'filename': all_filenames,
        'embedding_index': range(len(all_filenames))
    })
    metadata_df.to_csv(output_path_csv, index=False)

    print(f"Total images processed: {len(all_filenames)}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Saved files:")
    print(f"   - Embeddings: {output_path_npy}")
    print(f"   - Metadata: {output_path_csv}")

    # Копируем результаты в Google Drive
    drive_output_path = f"files/{output_filename}.npy"
    drive_csv_path = f"files/{output_filename}_metadata.csv"

    # Копируем файлы в Google Drive
    import shutil
    shutil.copy(output_path_npy, drive_output_path)
    shutil.copy(output_path_csv, drive_csv_path)

    return all_embeddings, all_filenames


IMAGE_DIR = "dataset/Объекты недвижимости, не соответствующие градостроительным нормам_00-022_Август"

print("Starting full dataset processing...")

embeddings, filenames = create_embeddings_colab(
    image_dir=IMAGE_DIR,
    output_filename="moscow_embeddings_512",
    batch_size=32,
    embedding_dim=512,
    max_images=500000
)


class FAISSHNSWSearch:
    def __init__(self, dimension, max_elements=100000):
        self.dimension = dimension
        self.max_elements = max_elements
        self.index = None
        self.embeddings = None
        self.filenames = []

    def create_index(self, M=16, efConstruction=200, efSearch=50):
        """Создание HNSW индекса в FAISS"""

        # Создаем HNSW индекс
        self.index = faiss.IndexHNSWFlat(self.dimension, M)

        # Устанавливаем параметры строительства
        self.index.hnsw.efConstruction = efConstruction
        self.index.efSearch = efSearch

        print(f"Создан FAISS HNSW индекс с M={M}, efConstruction={efConstruction}")

    def add_data(self, data, filenames=None):
        """Добавление данных в индекс"""
        if self.index is None:
            self.create_index()

        # Проверяем размерность данных
        if data.shape[1] != self.dimension:
            raise ValueError(f"Размерность данных {data.shape[1]} не совпадает с размерностью индекса {self.dimension}")

        # Добавляем данные
        self.index.add(data.astype('float32'))

        # Сохраняем эмбеддинги и имена файлов
        self.embeddings = data.astype('float32')
        if filenames is not None:
            self.filenames = filenames

        print(f"Добавлено {len(data)} векторов в индекс")

    def search(self, query, k=5, efSearch=None):
        """Поиск k ближайших соседей"""
        if self.index is None:
            raise ValueError("Индекс не создан!")

        if self.index.ntotal == 0:
            raise ValueError("Индекс пуст!")

        # Устанавливаем параметр поиска если указан
        if efSearch is not None:
            self.index.efSearch = efSearch

        # Проверяем и преобразуем query
        if isinstance(query, np.ndarray):
            if query.dtype != np.float32:
                query = query.astype('float32')
            if len(query.shape) == 1:
                query = query.reshape(1, -1)
        else:
            query = np.array(query, dtype='float32').reshape(1, -1)

        # Выполняем поиск
        start_time = time.time()
        distances, labels = self.index.search(query, k)
        search_time = (time.time() - start_time) * 1000  # в миллисекундах

        return labels[0], distances[0], search_time

    def search_batch(self, queries, k=5, efSearch=None):
        """Поиск для батча запросов"""
        if self.index is None:
            raise ValueError("Индекс не создан!")

        if efSearch is not None:
            self.index.efSearch = efSearch

        queries = queries.astype('float32')

        start_time = time.time()
        distances, labels = self.index.search(queries, k)
        search_time = (time.time() - start_time) * 1000

        return labels, distances, search_time

    def get_index_stats(self):
        """Получение статистики индекса"""
        if self.index is None:
            return "Индекс не создан"

        stats = {
            'total_vectors': self.index.ntotal,
            'dimension': self.index.d,
            'efSearch': self.index.efSearch,
            'M': self.index.hnsw.M,
            'efConstruction': self.index.hnsw.efConstruction,
            'max_level': self.index.hnsw.max_level,
            'entry_point': self.index.hnsw.entry_point
        }
        return stats

    def save_index(self, filename):
        """Сохранение индекса на диск"""
        if self.index is None:
            raise ValueError("Индекс не создан!")

        faiss.write_index(self.index, filename + '.bin')

        # Сохраняем метаданные
        metadata = {
            'filenames': self.filenames,
            'embeddings_shape': self.embeddings.shape if self.embeddings is not None else None
        }
        np.save(filename + '_metadata.npy', metadata)

        print(f"Индекс сохранен в {filename}.bin")
        print(f"Метаданные сохранены в {filename}_metadata.npy")

    def load_index(self, filename, load_embeddings=False):
        """Загрузка индекса с диска"""
        self.index = faiss.read_index(filename + '.bin')

        # Загружаем метаданные
        try:
            metadata = np.load(filename + '_metadata.npy', allow_pickle=True).item()
            self.filenames = metadata.get('filenames', [])
            if load_embeddings and metadata.get('embeddings_shape') is not None:
                # Для полной загрузки эмбеддингов нужно сохранять их отдельно
                print("Для загрузки эмбеддингов используйте отдельный файл")
        except FileNotFoundError:
            print("Метаданные не найдены, загружен только индекс")

        print(f"Индекс загружен из {filename}.bin")
        print(f"Загружено {self.index.ntotal} векторов")

    def set_ef_search(self, efSearch):
        """Установка параметра efSearch"""
        if self.index is None:
            raise ValueError("Индекс не создан!")
        self.index.efSearch = efSearch
        print(f"efSearch установлен в {efSearch}")


# Инициализация Flask приложения
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Глобальные переменные
model = None
faiss_searcher = None
filenames = []


def initialize_system():
    """Инициализация модели и загрузка индекса"""
    global model, faiss_searcher, filenames

    print("Инициализация системы поиска изображений...")

    # Загрузка модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OptimizedResNetEmbeddingExtractor(embedding_dim=512)
    model = model.to(device)
    model.eval()
    print(f"Модель загружена на устройство: {device}")

    # Загрузка эмбеддингов и метаданных
    try:
        # Пробуем несколько возможных путей
        possible_paths = [
            "moscow_embeddings.npy",
            "files/moscow_embeddings.npy",
            "/files/moscow_embeddings.npy",
            "embeddings.npy"
        ]

        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break

        if data_path is None:
            print("Файл с эмбеддингами не найден. Запуск в демо-режиме.")
            filenames = ["demo_image_1.jpg", "demo_image_2.jpg", "demo_image_3.jpg"]
            return

        print(f"Загружаем эмбеддинги из: {data_path}")
        data = np.load(data_path, allow_pickle=True).item()

        embeddings = data['embeddings']
        filenames = data['filenames']

        # Инициализация FAISS HNSW индекса
        print("Создание FAISS HNSW индекса...")
        faiss_searcher = FAISSHNSWSearch(dimension=512, max_elements=len(embeddings))
        faiss_searcher.add_data(embeddings, filenames)

        # Показываем статистику
        stats = faiss_searcher.get_index_stats()
        print(f"Статистика индекса: {stats}")

        print(f"✅ Система инициализирована. Загружено {len(filenames)} изображений")

    except Exception as e:
        print(f"❌ Ошибка при загрузке индекса: {e}")
        filenames = ["demo_image_1.jpg", "demo_image_2.jpg", "demo_image_3.jpg"]


def preprocess_image(image):
    """Предобработка изображения для модели"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, bytes):
        image = Image.open(io.BytesIO(image)).convert('RGB')

    return transform(image).unsqueeze(0)


def get_image_base64(image_path):
    """Конвертация изображения в base64 для отображения в HTML"""
    try:
        if os.path.exists(image_path):
            with open(image_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        else:
            # Пробуем найти файл в разных местах
            possible_paths = [
                image_path,
                os.path.join('dataset', os.path.basename(image_path)),
                os.path.join('/content', image_path),
                os.path.join('/content/drive/MyDrive', image_path)
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'rb') as img_file:
                        return base64.b64encode(img_file.read()).decode('utf-8')
            return ""
    except Exception as e:
        print(f"Ошибка при чтении изображения {image_path}: {e}")
        return ""


# Маршруты Flask
@app.route('/')
def index():
    """Главная страница с информацией"""
    stats = faiss_searcher.get_index_stats() if faiss_searcher else {}
    total_images = stats.get('total_vectors', 0) if isinstance(stats, dict) else 0

    return render_template('index.html',
                           total_images=total_images,
                           model_ready=model is not None,
                           index_stats=stats)


@app.route('/search')
def search_page():
    """Страница поиска"""
    return render_template('search.html')


@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint для поиска похожих изображений"""
    if model is None:
        return jsonify({'error': 'Модель не инициализирована'}), 500

    if faiss_searcher is None or faiss_searcher.index.ntotal == 0:
        return jsonify({'error': 'Поисковый индекс не загружен'}), 500

    try:
        # Проверяем загружен ли файл
        if 'image' not in request.files:
            return jsonify({'error': 'Файл изображения не предоставлен'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Изображение не выбрано'}), 400

        # Проверяем тип файла
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            return jsonify({'error': 'Неподдерживаемый формат изображения'}), 400

        # Читаем и обрабатываем изображение
        image_bytes = file.read()

        # Проверяем размер файла
        if len(image_bytes) == 0:
            return jsonify({'error': 'Файл пустой'}), 400

        input_tensor = preprocess_image(image_bytes)

        # Получаем эмбеддинг
        with torch.no_grad():
            device = next(model.parameters()).device
            input_tensor = input_tensor.to(device)
            query_embedding = model(input_tensor).cpu().numpy()

        # Ищем похожие изображения
        k = min(int(request.form.get('k', 5)), 20)  # максимум 20 результатов
        ef_search = request.form.get('ef_search')

        search_params = {}
        if ef_search:
            search_params['efSearch'] = int(ef_search)

        labels, distances, search_time = faiss_searcher.search(query_embedding, k=k, **search_params)

        # Формируем результаты
        results = []
        for i, (label, distance) in enumerate(zip(labels, distances)):
            if 0 <= label < len(faiss_searcher.filenames):
                image_path = faiss_searcher.filenames[label]
                image_base64 = get_image_base64(image_path)

                results.append({
                    'rank': i + 1,
                    'filename': os.path.basename(image_path),
                    'filepath': image_path,
                    'distance': float(distance),
                    'image_base64': image_base64,
                    'label_index': int(label)
                })

        # Статистика индекса
        index_stats = faiss_searcher.get_index_stats()

        return jsonify({
            'success': True,
            'results': results,
            'search_time_ms': search_time,
            'total_found': len(results),
            'query_dimension': query_embedding.shape[1],
            'index_stats': index_stats,
            'search_params': {
                'k': k,
                'ef_search': ef_search or index_stats.get('efSearch', 'default')
            }
        })

    except Exception as e:
        print(f"Ошибка при поиске: {e}")
        return jsonify({'error': f'Ошибка при обработке запроса: {str(e)}'}), 500


@app.route('/api/system_info')
def system_info():
    """API для получения информации о системе"""
    stats = faiss_searcher.get_index_stats() if faiss_searcher else {}

    info = {
        'model_loaded': model is not None,
        'index_loaded': faiss_searcher is not None and faiss_searcher.index.ntotal > 0,
        'total_images': stats.get('total_vectors', 0) if isinstance(stats, dict) else 0,
        'embedding_dimension': stats.get('dimension', 0) if isinstance(stats, dict) else 0,
        'index_type': 'FAISS HNSW',
        'device': str(next(model.parameters()).device) if model else 'unknown'
    }

    return jsonify(info)


@app.route('/api/image/<path:filename>')
def serve_image(filename):
    """Сервис для отдачи изображений"""
    try:
        # Безопасная проверка пути
        safe_filename = os.path.basename(filename)
        possible_paths = [
            os.path.join('dataset', safe_filename),
            os.path.join('/content', safe_filename),
            os.path.join('/content/drive/MyDrive', safe_filename),
            safe_filename
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return send_file(path)

        return "Image not found", 404
    except Exception as e:
        return f"Error serving image: {str(e)}", 500


@app.route('/api/reindex', methods=['POST'])
def reindex():
    """API для переиндексации (административная функция)"""
    try:
        # Проверка пароля или токена для безопасности
        auth_token = request.headers.get('Authorization')
        if auth_token != 'Bearer your-secret-token':  # Замените на реальный токен
            return jsonify({'error': 'Unauthorized'}), 401

        initialize_system()
        return jsonify({'success': True, 'message': 'Система переинициализирована'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Обработчики ошибок
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'Файл слишком большой'}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Страница не найдена'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Внутренняя ошибка сервера'}), 500


if __name__ == '__main__':
    # Инициализируем систему при запуске
    print("Запуск системы поиска изображений...")
    initialize_system()

    # Запускаем Flask приложение
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    print(f"Сервер запускается на порту {port}")
    print(f"Режим отладки: {debug}")
    print("Доступные маршруты:")
    print("   - GET  / (главная страница)")
    print("   - GET  /search (страница поиска)")
    print("   - POST /api/search (поиск похожих изображений)")
    print("   - GET  /api/system_info (информация о системе)")
    print("   - GET  /api/image/<filename> (получение изображения)")

    app.run(host='0.0.0.0', port=port, debug=debug)