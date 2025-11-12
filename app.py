import streamlit as st
import pandas as pd
from movie_recommender import MovieRecommender
import time

# Настройка страницы
st.set_page_config(
    page_title="🎬 Умная рекомендательная система фильмов",
    page_icon="🎬",
    layout="wide"
)

# Заголовок приложения
st.title("🎬 Умная рекомендательная система фильмов")
st.markdown("Система использует машинное обучение для предсказания рейтингов и рекомендаций фильмов")
st.markdown("---")

# Инициализация рекомендательной системы с обученной моделью
@st.cache_resource
def load_recommender():
    return MovieRecommender(
        data_path='kp_all_movies_cleanedd.csv',
        rating_model_path='trained_movie_model.pkl'  # Путь к вашей обученной модели
    )

# Загрузка данных
try:
    recommender = load_recommender()
    st.success("✅ Данные и модель машинного обучения успешно загружены!")
except Exception as e:
    st.error(f"❌ Ошибка загрузки данных или модели: {e}")
    st.stop()

# Боковая панель
st.sidebar.title("🔍 Настройки поиска")
st.sidebar.markdown("---")

# Поиск фильма
movie_query = st.sidebar.text_input(
    "Введите название фильма:",
    placeholder="Например: Матрица или The Matrix"
)

# Количество рекомендаций
n_recommendations = st.sidebar.slider(
    "Количество рекомендаций:",
    min_value=3,
    max_value=10,
    value=5
)

# Опция предсказания рейтингов
predict_ratings = st.sidebar.checkbox(
    "🎯 Использовать AI для предсказания рейтингов", 
    value=True,
    help="Система предскажет рейтинг фильма с помощью обученной модели машинного обучения"
)

# Кнопка поиска
search_button = st.sidebar.button("🎬 Найти похожие фильмы")

# Основная область
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🤖 О системе AI")
    st.info("""
    **Возможности системы:**
    
    🎯 **Рекомендации на основе:**
    - Жанров и тегов
    - Года выпуска
    - Рейтингов КП/IMDB
    - Страны производства
    
    🤖 **AI-функции:**
    - Предсказание рейтингов
    - Анализ схожести фильмов
    - Машинное обучение на исторических данных
    """)
    
    # Показать случайные фильмы
    if st.button("🎲 Случайные фильмы из базы"):
        with st.spinner("Ищем интересные фильмы..."):
            time.sleep(1)
            random_movies = recommender.get_random_movies(5)
            
            st.subheader("🎲 Случайные фильмы:")
            for i, movie in enumerate(random_movies, 1):
                with st.expander(f"{i}. {movie['name_rus']}"):
                    st.write(f"**Английское название:** {movie.get('name_eng', 'Нет данных')}")
                    st.write(f"**Жанры:** {movie.get('genres', 'Нет данных')}")
                    st.write(f"**Страны:** {movie.get('countries', 'Нет данных')}")
                    st.write(f"**Год:** {movie.get('movie_year', 'Нет данных')}")
                    st.write(f"**Рейтинг КП:** {movie.get('kp_rating', 'Нет данных')}")

with col2:
    st.subheader("🎯 Рекомендации фильмов")
    
    if search_button and movie_query:
        with st.spinner("🔍 Анализируем фильмы с помощью AI..."):
            time.sleep(1)
            original_movie, recommendations = recommender.find_similar_movies(
                movie_query, 
                n_recommendations,
                predict_ratings=predict_ratings
            )
            
            if recommendations is None:
                st.error(f"❌ {original_movie}")
            else:
                # Показать исходный фильм
                st.success(f"✅ Найден фильм: **{original_movie['name_rus']}**")
                
                with st.expander("📋 Информация о выбранном фильме", expanded=True):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Русское название:** {original_movie['name_rus']}")
                        st.write(f"**Английское название:** {original_movie.get('name_eng', 'Нет данных')}")
                        st.write(f"**Жанры:** {original_movie.get('genres', 'Нет данных')}")
                    with col_b:
                        st.write(f"**Страны:** {original_movie.get('countries', 'Нет данных')}")
                        st.write(f"**Год:** {original_movie.get('year', 'Нет данных')}")
                        st.write(f"**Рейтинг КП:** {original_movie.get('kp_rating', 'Нет данных')}")
                
                st.markdown("---")
                st.subheader("🎬 Рекомендуемые фильмы:")
                
                # Показать рекомендации
                for i, movie in enumerate(recommendations, 1):
                    # Создаем красивый заголовок с рейтингами
                    rating_info = ""
                    if predict_ratings and movie.get('predicted_rating'):
                        rating_info = f" | 🎯 AI рейтинг: {movie['predicted_rating']:.1f}"
                    
                    with st.expander(
                        f"{i}. {movie['name_rus']} "
                        f"(сходство: {movie['similarity']:.2f}{rating_info})", 
                        expanded=True
                    ):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**📊 Основная информация:**")
                            st.write(f"🎭 {movie.get('name_eng', 'Нет данных')}")
                            st.write(f"📅 {movie.get('year', 'Нет данных')} год")
                            st.write(f"⏱️ {movie.get('duration', 'Нет данных')} мин")
                        
                        with col2:
                            st.write("**⭐ Рейтинги:**")
                            st.write(f"🎬 КиноПоиск: {movie.get('kp_rating', 'Нет данных')}")
                            st.write(f"🎬 IMDB: {movie.get('imdb_rating', 'Нет данных')}")
                            if predict_ratings and movie.get('predicted_rating'):
                                st.write(f"🤖 AI предсказание: **{movie['predicted_rating']:.1f}**")
                            st.write(f"📊 Сходство: {movie['similarity']:.2f}")
                        
                        with col3:
                            st.write("**🎞️ Детали:**")
                            st.write(f"🎭 {movie.get('genres', 'Нет данных')}")
                            st.write(f"🌍 {movie.get('countries', 'Нет данных')}")
    
    elif not movie_query and search_button:
        st.warning("⚠️ Пожалуйста, введите название фильма для поиска")

# Футер
st.markdown("---")
st.markdown(
    "🎬 **Умная система рекомендаций фильмов** | "
    "AI на основе машинного обучения | "
    "© 2025"
)