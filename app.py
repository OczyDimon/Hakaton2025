import os
import numpy as np
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import io
import base64
import faiss
import time
import pandas as pd


def load_coordinates_table(csv_path=None):
    """Загрузка таблицы с координатами из XLSX файла"""
    global coordinates_df

    try:
        # Ищем XLSX файл
        xlsx_files = [
            "coordinates.xlsx",
            "files/coordinates.xlsx",
            "data.xlsx",
            "files/data.xlsx"
        ]

        xlsx_path = None
        for path in xlsx_files:
            if os.path.exists(path):
                xlsx_path = path
                print(f"✅ Найден файл с координатами: {path}")
                break

        if xlsx_path is None:
            print("❌ Файл с координатами не найден")
            coordinates_df = None
            return None

        # Загружаем XLSX файл
        coordinates_df = pd.read_excel(xlsx_path)
        print(f"📊 Загружена таблица координат: {len(coordinates_df)} записей")
        print(f"📋 Колонки: {list(coordinates_df.columns)}")

        return coordinates_df

    except Exception as e:
        print(f"❌ Ошибка загрузки XLSX: {e}")
        coordinates_df = None
        return None


def get_coordinates_by_filename(filename):
    """Получение координат по имени файла из таблицы"""
    global coordinates_df

    if coordinates_df is None:
        return None, None

    try:
        # Получаем только имя файла без пути
        basename = os.path.basename(filename)

        # Ищем в колонке "Имя файла"
        if 'Имя файла' in coordinates_df.columns:
            # Ищем точное совпадение
            match = coordinates_df[coordinates_df['Имя файла'] == basename]

            if not match.empty:
                row = match.iloc[0]

                # Берем координаты
                lon = row.get('longitude')
                lat = row.get('latitude')

                # Проверяем что значения не пустые
                if pd.notna(lon) and pd.notna(lat):
                    return float(lon), float(lat)

        return None, None

    except Exception as e:
        print(f"❌ Ошибка при поиске координат для {filename}: {e}")
        return None, None


# Глобальная переменная для таблицы координат
coordinates_df = None


# Класс для извлечения эмбеддингов
class OptimizedResNetEmbeddingExtractor(torch.nn.Module):
    def __init__(self, embedding_dim=512):
        super(OptimizedResNetEmbeddingExtractor, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='IMAGENET1K_V1')
        self.feature_extractor = torch.nn.Sequential(*list(self.backbone.children())[:-2])
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.projection = torch.nn.Linear(2048, embedding_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        pooled = self.global_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        embeddings = self.projection(flattened)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


# Класс для поиска с FAISS HNSW
class FAISSHNSWSearch:
    def __init__(self, dimension, max_elements=100000):
        self.dimension = dimension
        self.max_elements = max_elements
        self.index = None
        self.embeddings = None
        self.filenames = []

    def create_index(self, M=16, efConstruction=200, efSearch=50):
        """Создание HNSW индекса в FAISS"""
        self.index = faiss.IndexHNSWFlat(self.dimension, M)
        self.index.hnsw.efConstruction = efConstruction
        self.index.efSearch = efSearch
        print(f"Создан FAISS HNSW индекс с M={M}, efConstruction={efConstruction}")

    def add_data(self, data, filenames=None):
        """Добавление данных в индекс"""
        if self.index is None:
            self.create_index()

        if data.shape[1] != self.dimension:
            raise ValueError(f"Размерность данных {data.shape[1]} не совпадает с размерностью индекса {self.dimension}")

        self.index.add(data.astype('float32'))
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

        if efSearch is not None:
            self.index.efSearch = efSearch

        if isinstance(query, np.ndarray):
            if query.dtype != np.float32:
                query = query.astype('float32')
            if len(query.shape) == 1:
                query = query.reshape(1, -1)
        else:
            query = np.array(query, dtype='float32').reshape(1, -1)

        start_time = time.time()
        distances, labels = self.index.search(query, k)
        search_time = (time.time() - start_time) * 1000

        return labels[0], distances[0], search_time

    def get_index_stats(self):
        """Получение статистики индекса"""
        if self.index is None:
            return "Индекс не создан"

        # БЕЗОПАСНЫЙ ДОСТУП К ПАРАМЕТРАМ
        stats = {
            'total_vectors': self.index.ntotal,
            'dimension': self.index.d,
            'efSearch': getattr(self.index, 'efSearch', 'N/A'),
        }

        # ПРАВИЛЬНЫЙ ДОСТУП К HNSW ПАРАМЕТРАМ
        if hasattr(self.index, 'hnsw'):
            hnsw_obj = self.index.hnsw
            # Используем getattr для безопасного доступа
            stats['M'] = getattr(hnsw_obj, 'M', getattr(hnsw_obj, 'efConstruction', 'N/A'))
            stats['efConstruction'] = getattr(hnsw_obj, 'efConstruction', 'N/A')
            stats['max_level'] = getattr(hnsw_obj, 'max_level', 'N/A')
            stats['entry_point'] = getattr(hnsw_obj, 'entry_point', 'N/A')
        else:
            stats['M'] = 'N/A'
            stats['efConstruction'] = 'N/A'
            stats['max_level'] = 'N/A'
            stats['entry_point'] = 'N/A'

        return stats

    def save_index(self, filename):
        """Сохранение индекса на диск"""
        if self.index is None:
            raise ValueError("Индекс не создан!")

        # Создаем папку files если её нет
        os.makedirs('files', exist_ok=True)
        filepath = os.path.join('files', filename + '.bin')

        faiss.write_index(self.index, filepath)
        metadata = {
            'filenames': self.filenames,
            'embeddings_shape': self.embeddings.shape if self.embeddings is not None else None
        }

        metadata_path = os.path.join('files', filename + '_metadata.npy')
        np.save(metadata_path, metadata)
        print(f"Индекс сохранен в {filepath}")

    def load_index(self, filename):
        """Загрузка индекса с диска"""
        filepath = os.path.join('files', filename + '.bin')
        self.index = faiss.read_index(filepath)

        try:
            metadata_path = os.path.join('files', filename + '_metadata.npy')
            metadata = np.load(metadata_path, allow_pickle=True).item()
            self.filenames = metadata.get('filenames', [])
        except FileNotFoundError:
            print("Метаданные не найдены, загружен только индекс")

        print(f"Индекс загружен из {filepath}")
        print(f"Загружено {self.index.ntotal} векторов")


# Инициализация Flask приложения
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'files/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Создаем необходимые папки
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('files', exist_ok=True)

# Глобальные переменные
model = None
faiss_searcher = None
filenames = []


def initialize_system(embeddings_path=None):
    """Инициализация модели и загрузка индекса"""
    global model, faiss_searcher, filenames, coordinates_df

    print("Инициализация системы поиска изображений...")

    # Загрузка модели на CPU
    device = torch.device('cpu')
    model = OptimizedResNetEmbeddingExtractor(embedding_dim=512)
    model = model.to(device)
    model.eval()
    print("Модель загружена на CPU")

    # ЗАГРУЗКА ТАБЛИЦЫ КООРДИНАТ
    coordinates_df = load_coordinates_table()

    # Загрузка эмбеддингов и метаданных
    try:
        if embeddings_path is None:
            # Локальные пути
            possible_paths = [
                "files/embeddings.npy",
                "embeddings.npy",
                "files/moscow_embeddings_512.npy",
                "moscow_embeddings.npy"
            ]

            embeddings_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    embeddings_path = path
                    print(f"Найден файл эмбеддингов: {path}")
                    break

        if embeddings_path is None:
            print("Файл с эмбеддингами не найден. Запуск в демо-режиме.")
            # Создаем демо-индекс с случайными данными
            faiss_searcher = FAISSHNSWSearch(dimension=512, max_elements=1000)
            demo_embeddings = np.random.rand(50, 512).astype('float32')
            demo_filenames = [f"files/demo_{i}.jpg" for i in range(50)]
            faiss_searcher.add_data(demo_embeddings, demo_filenames)
            filenames = demo_filenames
            print("✅ Демо-система инициализирована")
            return

        print(f"Загружаем эмбеддинги из: {embeddings_path}")
        data = np.load(embeddings_path, allow_pickle=True).item()

        embeddings = data['embeddings']
        filenames = data['filenames']

        # Инициализация FAISS HNSW индекса
        print("Создание FAISS HNSW индекса...")
        faiss_searcher = FAISSHNSWSearch(dimension=512, max_elements=len(embeddings))
        faiss_searcher.add_data(embeddings, filenames)

        # Сохраняем индекс в папку files
        faiss_searcher.save_index("search_index")

        # Показываем статистику
        stats = faiss_searcher.get_index_stats()
        print(f"Статистика индекса: {stats}")

        print(f"✅ Система инициализирована. Загружено {len(filenames)} изображений")

    except Exception as e:
        print(f"❌ Ошибка при загрузке индекса: {e}")
        # Создаем демо-режим при ошибке
        faiss_searcher = FAISSHNSWSearch(dimension=512, max_elements=1000)
        demo_embeddings = np.random.rand(20, 512).astype('float32')
        demo_filenames = [f"files/demo_{i}.jpg" for i in range(20)]
        faiss_searcher.add_data(demo_embeddings, demo_filenames)
        filenames = demo_filenames
        print("✅ Демо-система инициализирована (режим ошибки)")


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
        # Если путь абсолютный и файл существует
        if os.path.exists(image_path):
            with open(image_path, 'rb') as img_file:
                image_data = img_file.read()
                return base64.b64encode(image_data).decode('utf-8')

        # Пробуем найти файл в разных местах
        filename_only = os.path.basename(image_path)

        # Список возможных путей для поиска
        search_paths = [
            os.path.join('files', filename_only),
            os.path.join('dataset/Объекты недвижимости, не соответствующие градостроительным нормам_00-022_Август', filename_only),
            filename_only,
            os.path.join('files', 'uploads', filename_only)
        ]

        for path in search_paths:
            if os.path.exists(path):
                with open(path, 'rb') as img_file:
                    image_data = img_file.read()
                    return base64.b64encode(image_data).decode('utf-8')

        print(f"⚠️ Изображение не найдено: {image_path}")
        return ""

    except Exception as e:
        print(f"❌ Ошибка при чтении изображения {image_path}: {e}")
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

        # Сохраняем файл во временную папку
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_search_image.jpg')
        file.save(temp_path)

        # Читаем и обрабатываем изображение
        with open(temp_path, 'rb') as f:
            image_bytes = f.read()

        input_tensor = preprocess_image(image_bytes)

        # Получаем эмбеддинг
        with torch.no_grad():
            device = next(model.parameters()).device
            input_tensor = input_tensor.to(device)
            query_embedding = model(input_tensor).cpu().numpy()

        # Ищем похожие изображения
        k = min(int(request.form.get('k', 5)), 20)
        labels, distances, search_time = faiss_searcher.search(query_embedding, k=k)

        # Формируем результаты
        results = []
        for i, (label, distance) in enumerate(zip(labels, distances)):
            if 0 <= label < len(faiss_searcher.filenames):
                image_path = faiss_searcher.filenames[label]
                image_base64 = get_image_base64(image_path)

                # ПОЛУЧАЕМ КООРДИНАТЫ
                longitude, latitude = get_coordinates_by_filename(image_path)

                results.append({
                    'rank': i + 1,
                    'filename': os.path.basename(image_path),
                    'filepath': image_path,
                    'distance': float(distance),
                    'image_base64': image_base64,
                    'label_index': int(label),
                    'longitude': longitude,
                    'latitude': latitude,
                    'has_coordinates': longitude is not None and latitude is not None
                })

        # Удаляем временный файл

        try:

            os.remove(temp_path)

        except:

            pass

        return jsonify({

            'success': True,

            'results': results,

            'search_time_ms': search_time,

            'total_found': len(results),

            'query_dimension': query_embedding.shape[1]

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
        'device': 'CPU'
    }

    return jsonify(info)


@app.route('/api/image/<filename>')
def serve_image(filename):
    """Сервис для отдачи изображений"""
    try:
        # Безопасная проверка пути
        safe_filename = os.path.basename(filename)

        # Пробуем разные пути
        possible_paths = [
            os.path.join('files', safe_filename),
            safe_filename,
            os.path.join('dataset/Объекты недвижимости, не соответствующие градостроительным нормам_00-022_Август', safe_filename)
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return send_file(path)

        return "Image not found", 404
    except Exception as e:
        return f"Error serving image: {str(e)}", 500


@app.route('/api/upload_index', methods=['POST'])
def upload_index():
    """API для загрузки файла с эмбеддингами"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Файл не предоставлен'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400

        if not file.filename.endswith('.npy'):
            return jsonify({'error': 'Файл должен быть в формате .npy'}), 400

        # Сохраняем файл в папку files
        filepath = os.path.join('files', 'embeddings.npy')
        file.save(filepath)

        # Переинициализируем систему
        initialize_system(filepath)

        return jsonify({
            'success': True,
            'message': 'Файл эмбеддингов успешно загружен и система переинициализирована'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_coordinates', methods=['POST'])
def upload_coordinates():
    """API для загрузки CSV файла с координатами"""
    global coordinates_df

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Файл не предоставлен'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Файл должен быть в формате .csv'}), 400

        # Сохраняем файл в папку files
        filepath = os.path.join('files', 'coordinates.csv')
        file.save(filepath)

        # Загружаем таблицу
        coordinates_df = load_coordinates_table(filepath)

        return jsonify({
            'success': True,
            'message': f'Таблица координатов успешно загружена. Записей: {len(coordinates_df) if coordinates_df is not None else 0}',
            'columns': list(coordinates_df.columns) if coordinates_df is not None else []
        })

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
    # Для Windows
    import multiprocessing

    multiprocessing.freeze_support()

    # Инициализируем систему при запуске
    print("🚀 Запуск системы поиска изображений...")
    initialize_system()

    # Запускаем Flask приложение
    port = 5000
    debug = True

    print(f"📍 Сервер запускается на http://localhost:{port}")
    print(f"📁 Все файлы сохраняются в папку 'files'")
    print("📖 Доступные маршруты:")
    print("   - GET  / (главная страница)")
    print("   - GET  /search (страница поиска)")
    print("   - POST /api/search (поиск похожих изображений)")
    print("   - GET  /api/system_info (информация о системе)")
    print("   - POST /api/upload_index (загрузка файла эмбеддингов)")

    app.run(host='0.0.0.0', port=port, use_reloader=False)