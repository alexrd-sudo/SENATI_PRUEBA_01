import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.impute import SimpleImputer
import base64
from datetime import datetime
import io

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🎓 Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# CSS PERSONALIZADO - DISEÑO INCREÍBLE
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Fondo principal con gradiente */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
        color: #e0e0e0;
    }

    /* Sidebar */
    .css-1d391kg, .css-1cpxqw2 {
        background: linear-gradient(180deg, #1e1e3f 0%, #0f0c29 100%);
    }

    /* Títulos */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
    }

    /* Tarjetas métricas */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #a0a0c0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Botones */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }

    /* Selectbox y inputs */
    .stSelectbox > div > div, .stNumberInput > div > div, .stSlider > div > div {
        background: rgba(30, 30, 63, 0.8);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        color: white;
    }

    /* Dataframes */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Alertas personalizadas */
    .success-box {
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.15), rgba(56, 161, 105, 0.15));
        border-left: 4px solid #48bb78;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0;
    }

    .info-box {
        background: linear-gradient(135deg, rgba(66, 153, 225, 0.15), rgba(49, 130, 206, 0.15));
        border-left: 4px solid #4299e1;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0;
    }

    .warning-box {
        background: linear-gradient(135deg, rgba(237, 137, 54, 0.15), rgba(221, 107, 32, 0.15));
        border-left: 4px solid #ed8936;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0;
    }

    /* Hero section */
    .hero-section {
        text-align: center;
        padding: 40px 20px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: 24px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin-bottom: 30px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 30, 63, 0.6);
        border-radius: 12px 12px 0 0;
        padding: 12px 24px;
        color: #a0a0c0;
        font-weight: 600;
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-bottom: none;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.3));
        color: white;
        border-color: rgba(102, 126, 234, 0.5);
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 30, 63, 0.6);
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }

    /* Checkbox */
    .stCheckbox > label {
        color: #c0c0e0;
    }

    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        margin: 30px 0;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1a1a3e;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 4px;
    }

    /* Plotly container */
    .js-plotly-plot .plotly .modebar {
        display: none !important;
    }

    /* Sidebar header */
    .sidebar-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        border-radius: 16px;
        margin-bottom: 20px;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #6060a0;
        font-size: 0.85rem;
        border-top: 1px solid rgba(102, 126, 234, 0.2);
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# GENERACIÓN DE BASE DE DATOS GRANDE (5000+ registros)
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def generate_large_dataset(n_samples=5000):
    """Genera un dataset grande y realista de rendimiento estudiantil"""
    np.random.seed(42)

    data = {
        'student_id': range(1, n_samples + 1),
        'age': np.random.randint(16, 26, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'socioeconomic_level': np.random.choice(['Bajo', 'Medio', 'Alto'], n_samples, p=[0.3, 0.5, 0.2]),
        'parent_education': np.random.choice(
            ['Sin estudios', 'Primaria', 'Secundaria', 'Técnico', 'Universitario', 'Posgrado'],
            n_samples, p=[0.05, 0.1, 0.25, 0.2, 0.3, 0.1]
        ),
        'study_hours_weekly': np.random.normal(15, 6, n_samples).clip(1, 40).astype(int),
        'study_hours_daily': np.random.normal(2.5, 1.2, n_samples).clip(0.5, 8).round(1),
        'attendance_rate': np.random.normal(82, 12, n_samples).clip(40, 100).round(1),
        'homework_completion': np.random.normal(75, 15, n_samples).clip(20, 100).round(1),
        'participation_score': np.random.normal(70, 18, n_samples).clip(10, 100).round(1),
        'midterm_exam_score': np.random.normal(65, 18, n_samples).clip(0, 100).round(1),
        'quiz_average': np.random.normal(68, 16, n_samples).clip(0, 100).round(1),
        'project_score': np.random.normal(72, 15, n_samples).clip(0, 100).round(1),
        'lab_score': np.random.normal(70, 17, n_samples).clip(0, 100).round(1),
        'previous_gpa': np.random.normal(3.0, 0.8, n_samples).clip(0, 4.0).round(2),
        'extracurricular_activities': np.random.randint(0, 5, n_samples),
        'internet_access': np.random.choice(['Sí', 'No'], n_samples, p=[0.85, 0.15]),
        'device_access': np.random.choice(['Computadora', 'Tablet', 'Celular', 'Ninguno'], n_samples, p=[0.5, 0.2, 0.25, 0.05]),
        'sleep_hours': np.random.normal(7, 1.5, n_samples).clip(3, 12).round(1),
        'stress_level': np.random.choice(['Bajo', 'Medio', 'Alto'], n_samples, p=[0.3, 0.45, 0.25]),
        'motivation_level': np.random.choice(['Bajo', 'Medio', 'Alto'], n_samples, p=[0.15, 0.45, 0.4]),
        'learning_style': np.random.choice(['Visual', 'Auditivo', 'Kinestésico', 'Lector'], n_samples),
        'course_difficulty': np.random.choice(['Baja', 'Media', 'Alta'], n_samples, p=[0.2, 0.5, 0.3]),
        'teacher_rating': np.random.normal(3.5, 0.8, n_samples).clip(1, 5).round(1),
        'peer_study_group': np.random.choice(['Sí', 'No'], n_samples, p=[0.6, 0.4]),
        'tutoring_sessions': np.random.poisson(2, n_samples),
        'absences': np.random.poisson(3, n_samples),
        'late_assignments': np.random.poisson(2, n_samples),
        'final_grade': None,
        'passed': None
    }

    df = pd.DataFrame(data)

    # Calcular nota final basada en múltiples factores (fórmula realista)
    base_score = (
        df['midterm_exam_score'] * 0.25 +
        df['quiz_average'] * 0.15 +
        df['project_score'] * 0.20 +
        df['lab_score'] * 0.15 +
        df['participation_score'] * 0.10 +
        df['homework_completion'] * 0.15
    )

    # Factores de ajuste
    adjustment = (
        (df['attendance_rate'] - 75) * 0.1 +
        (df['study_hours_weekly'] - 15) * 0.3 +
        (df['previous_gpa'] - 3.0) * 5 +
        np.where(df['motivation_level'] == 'Alto', 5, np.where(df['motivation_level'] == 'Medio', 0, -5)) +
        np.where(df['stress_level'] == 'Alto', -3, 0) +
        np.where(df['peer_study_group'] == 'Sí', 3, 0) +
        np.where(df['internet_access'] == 'Sí', 2, -3) +
        (df['tutoring_sessions'] - 2) * 1.5 -
        df['absences'] * 1.2 -
        df['late_assignments'] * 1.5
    )

    df['final_grade'] = (base_score + adjustment).clip(0, 100).round(1)

    # Agregar ruido realista
    noise = np.random.normal(0, 3, n_samples)
    df['final_grade'] = (df['final_grade'] + noise).clip(0, 100).round(1)

    df['passed'] = (df['final_grade'] >= 60).astype(int)

    # Agregar algunos valores nulos para simular datos reales (~3%)
    null_cols = ['study_hours_daily', 'participation_score', 'sleep_hours', 'stress_level']
    for col in null_cols:
        null_indices = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
        df.loc[null_indices, col] = np.nan

    return df


# ═══════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════
def create_metric_card(label, value, icon="📊"):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


def plot_confusion_matrix_matrix(cm, labels, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=2, linecolor='white',
                annot_kws={'size': 14, 'fontweight': 'bold'})
    ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=15)
    ax.set_xlabel('Predicción', fontsize=12, color='#c0c0e0')
    ax.set_ylabel('Real', fontsize=12, color='#c0c0e0')
    ax.tick_params(colors='#c0c0e0')
    fig.patch.set_facecolor('#1a1a3e')
    ax.set_facecolor('#24243e')
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {roc_auc:.4f})',
        line=dict(color='#667eea', width=3),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='#6060a0', width=2, dash='dash')
    ))
    fig.update_layout(
        title=f'Curva ROC - {model_name}',
        xaxis_title='Tasa de Falsos Positivos',
        yaxis_title='Tasa de Verdaderos Positivos',
        template='plotly_dark',
        paper_bgcolor='#1a1a3e',
        plot_bgcolor='#24243e',
        font=dict(color='#e0e0e0'),
        height=450
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2 style="margin:0; font-size:1.8rem;">🎓</h2>
        <h3 style="margin:5px 0; font-size:1.1rem; color:#667eea;">Student Analytics</h3>
        <p style="margin:0; font-size:0.75rem; color:#a0a0c0;">Rendimiento Académico</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navegación
    page = st.radio(
        "🧭 **Navegación**",
        ["🏠 Inicio", "📊 Exploración de Datos", "🧹 Preprocesamiento",
         "🤖 Parte A: Supervisado", "🔬 Parte B: No Supervisado",
         "🔮 Predicción Individual", "📋 Conclusiones"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Configuración del dataset
    st.subheader("⚙️ Configuración")
    dataset_size = st.select_slider(
        "Tamaño del dataset",
        options=[1000, 2000, 3000, 5000, 8000, 10000],
        value=5000
    )

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:10px;">
        <p style="color:#6060a0; font-size:0.75rem;">
            Proyecto de Machine Learning<br>
            Aprendizaje Supervisado & No Supervisado<br>
            © 2026 - SENATI
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# GENERAR DATASET
# ═══════════════════════════════════════════════════════════════
with st.spinner("Generando base de datos académica..."):
    df_raw = generate_large_dataset(dataset_size)

# ═══════════════════════════════════════════════════════════════
# PÁGINA: INICIO
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Inicio":
    st.markdown("""
    <div class="hero-section">
        <h1 style="font-size:3rem; margin-bottom:10px;">🎓 Sistema de Análisis de Rendimiento Académico</h1>
        <p style="font-size:1.2rem; color:#a0a0c0; max-width:700px; margin:0 auto;">
            Plataforma inteligente para analizar y predecir el rendimiento estudiantil
            utilizando técnicas de Machine Learning Supervisado y No Supervisado
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(create_metric_card("Total Estudiantes", f"{len(df_raw):,}", "👨‍🎓"), unsafe_allow_html=True)
    with col2:
        passed = df_raw['passed'].sum()
        st.markdown(create_metric_card("Aprobados", f"{passed:,} ({passed/len(df_raw)*100:.1f}%)", "✅"), unsafe_allow_html=True)
    with col3:
        failed = len(df_raw) - passed
        st.markdown(create_metric_card("Reprobados", f"{failed:,} ({failed/len(df_raw)*100:.1f}%)", "❌"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card("Nota Promedio", f"{df_raw['final_grade'].mean():.1f}", "📈"), unsafe_allow_html=True)

    st.markdown("---")

    # Descripción del proyecto
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🎯 Objetivos del Proyecto")
        st.markdown("""
        <div class="info-box">
            <strong>1. Aprendizaje Supervisado:</strong> Predecir si un estudiante aprobará o reprobará
            el curso de matemáticas utilizando algoritmos de clasificación (KNN y Random Forest).
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="success-box">
            <strong>2. Aprendizaje No Supervisado:</strong> Agrupar estudiantes según sus características
            académicas utilizando K-Means para identificar patrones de comportamiento.
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.subheader("📚 Tecnologías Utilizadas")
        tech_data = {
            "Categoría": ["Lenguaje", "Framework Web", "ML Supervisado", "ML No Supervisado", "Visualización", "Análisis de Datos"],
            "Tecnología": ["Python 3.11", "Streamlit", "KNN, Random Forest", "K-Means", "Plotly, Matplotlib, Seaborn", "Pandas, NumPy, Scikit-learn"]
        }
        tech_df = pd.DataFrame(tech_data)
        st.dataframe(tech_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Distribución de notas
    st.subheader("📊 Distribución de Notas Finales")
    fig_hist = px.histogram(
        df_raw, x='final_grade', nbins=50,
        color='passed', color_discrete_map={0: '#e74c3c', 1: '#2ecc71'},
        labels={'final_grade': 'Nota Final', 'count': 'Cantidad de Estudiantes', 'passed': 'Estado'},
        title='Distribución de Notas Finales'
    )
    fig_hist.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1a1a3e',
        plot_bgcolor='#24243e',
        font=dict(color='#e0e0e0', size=12),
        height=400,
        bargap=0.02
    )
    fig_hist.add_vline(x=60, line_dash="dash", line_color="#f39c12",
                       annotation_text="Umbral Aprobación (60)", annotation_position="top")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Distribución por género y nivel socioeconómico
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_gender = px.pie(
            df_raw, names='gender',
            title='Distribución por Género',
            color_discrete_sequence=['#667eea', '#764ba2']
        )
        fig_gender.update_layout(template='plotly_dark', paper_bgcolor='#1a1a3e',
                                  font=dict(color='#e0e0e0'), height=350)
        st.plotly_chart(fig_gender, use_container_width=True)

    with col_g2:
        fig_socio = px.pie(
            df_raw, names='socioeconomic_level',
            title='Nivel Socioeconómico',
            color_discrete_sequence=['#e74c3c', '#f39c12', '#2ecc71']
        )
        fig_socio.update_layout(template='plotly_dark', paper_bgcolor='#1a1a3e',
                                 font=dict(color='#e0e0e0'), height=350)
        st.plotly_chart(fig_socio, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PÁGINA: EXPLORACIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Exploración de Datos":
    st.header("📊 Exploración y Análisis de Datos")
    st.markdown("Análisis exhaustivo del dataset académico con **{}** registros y **{}** características.".format(
        len(df_raw), len(df_raw.columns)))

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Vista de Datos", "📈 Estadísticas", "🔍 Valores Nulos",
        "📊 Correlaciones", "🎨 Distribuciones"
    ])

    with tab1:
        st.subheader("Vista previa del dataset")
        st.dataframe(df_raw.head(20), use_container_width=True, height=400)

        st.markdown("---")
        st.subheader("Información del Dataset")
        buffer = io.StringIO()
        df_raw.info(buf=buffer)
        st.text(buffer.getvalue())

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(f"**Shape:** {df_raw.shape[0]} filas × {df_raw.shape[1]} columnas")
        with col_d2:
            st.markdown(f"**Memoria usada:** {df_raw.memory_usage(deep=True).sum() / 1024:.1f} KB")

    with tab2:
        st.subheader("Estadísticas Descriptivas")
        st.dataframe(df_raw.describe().round(2), use_container_width=True)

        st.markdown("---")
        st.subheader("Estadísticas por Variable Categórica")
        cat_cols = ['gender', 'socioeconomic_level', 'parent_education', 'stress_level',
                    'motivation_level', 'learning_style', 'course_difficulty',
                    'internet_access', 'device_access', 'peer_study_group']

        for col in cat_cols:
            with st.expander(f"📊 Distribución: {col.replace('_', ' ').title()}"):
                counts = df_raw[col].value_counts()
                fig = px.bar(
                    x=counts.index, y=counts.values,
                    color=counts.index,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    labels={'x': col.replace('_', ' ').title(), 'y': 'Cantidad'}
                )
                fig.update_layout(
                    template='plotly_dark', paper_bgcolor='#1a1a3e', plot_bgcolor='#24243e',
                    font=dict(color='#e0e0e0'), showlegend=False, height=350
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("🔍 Análisis de Valores Nulos")
        null_counts = df_raw.isnull().sum()
        null_df = pd.DataFrame({
            'Columna': null_counts.index,
            'Nulos': null_counts.values,
            'Porcentaje': (null_counts.values / len(df_raw) * 100).round(2)
        })
        null_df = null_df[null_df['Nulos'] > 0].sort_values('Nulos', ascending=False)

        if len(null_df) > 0:
            st.dataframe(null_df, use_container_width=True, hide_index=True)

            fig_null = px.bar(
                null_df, x='Columna', y='Porcentaje',
                color='Porcentaje',
                color_continuous_scale='Reds',
                title='Porcentaje de Valores Nulos por Columna'
            )
            fig_null.update_layout(
                template='plotly_dark', paper_bgcolor='#1a1a3e', plot_bgcolor='#24243e',
                font=dict(color='#e0e0e0'), height=400
            )
            st.plotly_chart(fig_null, use_container_width=True)
        else:
            st.success("✅ No se encontraron valores nulos en el dataset.")

    with tab4:
        st.subheader("📊 Matriz de Correlación")
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = df_raw[numeric_cols].corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title='Matriz de Correlación de Variables Numéricas',
            aspect='auto'
        )
        fig_corr.update_layout(
            template='plotly_dark', paper_bgcolor='#1a1a3e',
            font=dict(color='#e0e0e0', size=10), height=700
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("---")
        st.subheader("🔗 Correlaciones con la Nota Final")
        corr_with_grade = corr_matrix['final_grade'].drop('final_grade').sort_values(ascending=False)
        fig_corr_grade = px.bar(
            x=corr_with_grade.values, y=corr_with_grade.index,
            orientation='h',
            color=corr_with_grade.values,
            color_continuous_scale=['#e74c3c', '#f39c12', '#2ecc71'],
            title='Correlación de Variables con la Nota Final'
        )
        fig_corr_grade.update_layout(
            template='plotly_dark', paper_bgcolor='#1a1a3e', plot_bgcolor='#24243e',
            font=dict(color='#e0e0e0'), height=500,
            xaxis_title='Coeficiente de Correlación', yaxis_title='Variable'
        )
        st.plotly_chart(fig_corr_grade, use_container_width=True)

    with tab5:
        st.subheader("🎨 Distribuciones de Variables Numéricas")
        num_cols_plot = ['age', 'study_hours_weekly', 'attendance_rate', 'homework_completion',
                         'participation_score', 'midterm_exam_score', 'quiz_average',
                         'project_score', 'lab_score', 'previous_gpa', 'final_grade',
                         'sleep_hours', 'absences']

        cols_plot = st.columns(3)
        for i, col in enumerate(num_cols_plot):
            with cols_plot[i % 3]:
                fig = px.histogram(
                    df_raw, x=col, nbins=30,
                    color_discrete_sequence=['#667eea'],
                    title=f'Distribución: {col.replace("_", " ").title()}'
                )
                fig.update_layout(
                    template='plotly_dark', paper_bgcolor='#1a1a3e', plot_bgcolor='#24243e',
                    font=dict(color='#e0e0e0', size=10), height=280,
                    showlegend=False, margin=dict(t=40, b=30, l=40, r=20)
                )
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PÁGINA: PREPROCESAMIENTO
# ═══════════════════════════════════════════════════════════════
elif page == "🧹 Preprocesamiento":
    st.header("🧹 Limpieza y Preprocesamiento de Datos")

    st.markdown("""
    <div class="info-box">
        <strong>Proceso de preprocesamiento:</strong> Imputación de valores nulos, codificación de variables
        categóricas, escalado de características y división train/test.
    </div>
    """, unsafe_allow_html=True)

    # Paso 1: Manejo de nulos
    st.subheader("Paso 1: Manejo de Valores Nulos")
    df_clean = df_raw.copy()

    # Imputar nulos numéricos con la mediana
    numeric_null_cols = df_clean.select_dtypes(include=[np.number]).columns[df_clean.isnull().any()].tolist()
    for col in numeric_null_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Imputar nulos categóricos con la moda
    cat_null_cols = df_clean.select_dtypes(include=['object']).columns[df_clean.isnull().any()].tolist()
    for col in cat_null_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    st.success(f"✅ Valores nulos imputados: {len(numeric_null_cols)} columnas numéricas (mediana), {len(cat_null_cols)} categóricas (moda)")

    # Paso 2: Codificación
    st.subheader("Paso 2: Codificación de Variables Categóricas")

    le_gender = LabelEncoder()
    df_clean['gender_encoded'] = le_gender.fit_transform(df_clean['gender'])

    le_socio = LabelEncoder()
    df_clean['socioeconomic_encoded'] = le_socio.fit_transform(df_clean['socioeconomic_level'])

    le_parent = LabelEncoder()
    df_clean['parent_education_encoded'] = le_parent.fit_transform(df_clean['parent_education'])

    le_stress = LabelEncoder()
    df_clean['stress_encoded'] = le_stress.fit_transform(df_clean['stress_level'])

    le_motivation = LabelEncoder()
    df_clean['motivation_encoded'] = le_motivation.fit_transform(df_clean['motivation_level'])

    le_style = LabelEncoder()
    df_clean['learning_style_encoded'] = le_style.fit_transform(df_clean['learning_style'])

    le_difficulty = LabelEncoder()
    df_clean['difficulty_encoded'] = le_difficulty.fit_transform(df_clean['course_difficulty'])

    le_internet = LabelEncoder()
    df_clean['internet_encoded'] = le_internet.fit_transform(df_clean['internet_access'])

    le_device = LabelEncoder()
    df_clean['device_encoded'] = le_device.fit_transform(df_clean['device_access'])

    le_peer = LabelEncoder()
    df_clean['peer_encoded'] = le_peer.fit_transform(df_clean['peer_study_group'])

    st.success("✅ Variables categóricas codificadas con LabelEncoder")

    # Paso 3: Selección de features
    st.subheader("Paso 3: Selección de Características")

    feature_cols = [
        'age', 'gender_encoded', 'socioeconomic_encoded', 'parent_education_encoded',
        'study_hours_weekly', 'study_hours_daily', 'attendance_rate', 'homework_completion',
        'participation_score', 'midterm_exam_score', 'quiz_average', 'project_score',
        'lab_score', 'previous_gpa', 'extracurricular_activities', 'internet_encoded',
        'device_encoded', 'sleep_hours', 'stress_encoded', 'motivation_encoded',
        'learning_style_encoded', 'difficulty_encoded', 'teacher_rating',
        'peer_encoded', 'tutoring_sessions', 'absences', 'late_assignments'
    ]

    target_col = 'passed'

    X = df_clean[feature_cols]
    y = df_clean[target_col]

    st.markdown(f"**Features seleccionadas:** {len(feature_cols)}")
    st.markdown(f"**Variable objetivo:** `{target_col}` (0 = Reprobado, 1 = Aprobado)")

    # Paso 4: División train/test
    st.subheader("Paso 4: División Train/Test")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        test_size = st.slider("Porcentaje de Test", 10, 40, 20, 5)
    with col_t2:
        random_state = st.number_input("Random State", value=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=random_state, stratify=y
    )

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.markdown(create_metric_card("Total", f"{len(df_clean):,}", "📊"), unsafe_allow_html=True)
    with col_s2:
        st.markdown(create_metric_card("Train", f"{len(X_train):,} ({100-test_size}%)", "🏋️"), unsafe_allow_html=True)
    with col_s3:
        st.markdown(create_metric_card("Test", f"{len(X_test):,} ({test_size}%)", "🧪"), unsafe_allow_html=True)
    with col_s4:
        st.markdown(create_metric_card("Features", f"{len(feature_cols)}", "⚙️"), unsafe_allow_html=True)

    # Paso 5: Escalado
    st.subheader("Paso 5: Escalado de Características (StandardScaler)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.success("✅ Datos escalados con StandardScaler (media=0, desviación=1)")

    # Mostrar distribución antes/después del escalado
    st.subheader("Comparación: Antes vs Después del Escalado")
    col_ba, col_af = st.columns(2)

    with col_ba:
        fig_before = px.histogram(
            pd.DataFrame(X_train)['midterm_exam_score'],
            nbins=30, color_discrete_sequence=['#e74c3c'],
            title='Antes del Escalado'
        )
        fig_before.update_layout(template='plotly_dark', paper_bgcolor='#1a1a3e',
                                  plot_bgcolor='#24243e', font=dict(color='#e0e0e0'),
                                  showlegend=False, height=300,
                                  xaxis_title='midterm_exam_score')
        st.plotly_chart(fig_before, use_container_width=True)

    with col_af:
        fig_after = px.histogram(
            pd.DataFrame(X_train_scaled)[:, feature_cols.index('midterm_exam_score')],
            nbins=30, color_discrete_sequence=['#2ecc71'],
            title='Después del Escalado'
        )
        fig_after.update_layout(template='plotly_dark', paper_bgcolor='#1a1a3e',
                                 plot_bgcolor='#24243e', font=dict(color='#e0e0e0'),
                                 showlegend=False, height=300,
                                 xaxis_title='midterm_exam_score (escalado)')
        st.plotly_chart(fig_after, use_container_width=True)

    # Guardar datos procesados en session state
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test
    st.session_state['X_train_scaled'] = X_train_scaled
    st.session_state['X_test_scaled'] = X_test_scaled
    st.session_state['scaler'] = scaler
    st.session_state['feature_cols'] = feature_cols
    st.session_state['df_clean'] = df_clean


# ═══════════════════════════════════════════════════════════════
# PÁGINA: PARTE A - APRENDIZAJE SUPERVISADO
# ═══════════════════════════════════════════════════════════════
elif page == "🤖 Parte A: Supervisado":
    st.header("🤖 Parte A: Aprendizaje Supervisado")
    st.markdown("""
    <div class="info-box">
        <strong>Objetivo:</strong> Predecir si un estudiante aprobará (1) o reprobará (0) el curso de matemáticas.
        <br>Algoritmos: <strong>KNN</strong> y <strong>Random Forest</strong>
    </div>
    """, unsafe_allow_html=True)

    # Asegurar que los datos estén procesados
    if 'X_train_scaled' not in st.session_state:
        st.warning("⚠️ Primero debes completar el preprocesamiento. Ve a la sección '🧹 Preprocesamiento'.")
        st.stop()

    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train = st.session_state['y_train']
    y_test = st.session_state['y_test']
    X_train_scaled = st.session_state['X_train_scaled']
    X_test_scaled = st.session_state['X_test_scaled']
    feature_cols = st.session_state['feature_cols']

    tab_knn, tab_rf, tab_compare = st.tabs(["🔵 KNN", "🌲 Random Forest", "⚖️ Comparación"])

    # ───────────────────────────────────────────────────────────
    # KNN
    # ───────────────────────────────────────────────────────────
    with tab_knn:
        st.subheader("🔵 K-Nearest Neighbors (KNN)")

        col_k1, col_k2 = st.columns(2)
        with col_k1:
            k_neighbors = st.slider("Número de vecinos (k)", 3, 30, 7)
        with col_k2:
            weights = st.selectbox("Pesos", ['uniform', 'distance'])

        with st.spinner("Entrenando modelo KNN..."):
            knn = KNeighborsClassifier(n_neighbors=k_neighbors, weights=weights)
            knn.fit(X_train_scaled, y_train)
            knn_pred = knn.predict(X_test_scaled)
            knn_proba = knn.predict_proba(X_test_scaled)[:, 1]

        # Métricas KNN
        knn_acc = accuracy_score(y_test, knn_pred)
        knn_prec = precision_score(y_test, knn_pred)
        knn_rec = recall_score(y_test, knn_pred)
        knn_f1 = f1_score(y_test, knn_pred)

        st.markdown("---")
        st.subheader("📊 Métricas del Modelo KNN")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(create_metric_card("Accuracy", f"{knn_acc:.4f}", "🎯"), unsafe_allow_html=True)
        with col_m2:
            st.markdown(create_metric_card("Precision", f"{knn_prec:.4f}", "📌"), unsafe_allow_html=True)
        with col_m3:
            st.markdown(create_metric_card("Recall", f"{knn_rec:.4f}", "🔍"), unsafe_allow_html=True)
        with col_m4:
            st.markdown(create_metric_card("F1-Score", f"{knn_f1:.4f}", "⚡"), unsafe_allow_html=True)

        # Cross-validation
        cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
        st.markdown(f"**Cross-Validation (5 folds):** {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Confusion Matrix
        st.subheader("📋 Matriz de Confusión - KNN")
        cm_knn = confusion_matrix(y_test, knn_pred)
        fig_cm_knn = plot_confusion_matrix_matrix(cm_knn, ['Reprobado (0)', 'Aprobado (1)'], 'Matriz de Confusión - KNN')
        st.pyplot(fig_cm_knn)

        # Classification Report
        st.subheader("📝 Classification Report - KNN")
        st.text(classification_report(y_test, knn_pred, target_names=['Reprobado', 'Aprobado']))

        # ROC Curve
        st.subheader("📈 Curva ROC - KNN")
        fig_roc_knn = plot_roc_curve(y_test, knn_proba, 'KNN')
        st.plotly_chart(fig_roc_knn, use_container_width=True)

        # Feature Importance KNN (basado en correlación)
        st.subheader("🔑 Importancia de Features - KNN")
        feature_importance_knn = np.abs(X_train_scaled.T @ (y_train.values - y_train.mean()))
        importance_df_knn = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': feature_importance_knn
        }).sort_values('Importance', ascending=False).head(10)

        fig_imp_knn = px.bar(
            importance_df_knn, x='Importance', y='Feature',
            orientation='h', color='Importance',
            color_continuous_scale='Purples',
            title='Top 10 Features Más Importantes - KNN'
        )
        fig_imp_knn.update_layout(
            template='plotly_dark', paper_bgcolor='#1a1a3e', plot_bgcolor='#24243e',
            font=dict(color='#e0e0e0'), height=400
        )
        st.plotly_chart(fig_imp_knn, use_container_width=True)

    # ───────────────────────────────────────────────────────────
    # RANDOM FOREST
    # ───────────────────────────────────────────────────────────
    with tab_rf:
        st.subheader("🌲 Random Forest Classifier")

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            n_estimators = st.slider("Número de árboles", 50, 500, 100, 50)
        with col_r2:
            max_depth = st.slider("Profundidad máxima", 3, 30, 10)
        with col_r3:
            min_samples = st.slider("Min samples leaf", 1, 20, 5)

        with st.spinner("Entrenando modelo Random Forest..."):
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train_scaled, y_train)
            rf_pred = rf.predict(X_test_scaled)
            rf_proba = rf.predict_proba(X_test_scaled)[:, 1]

        # Métricas RF
        rf_acc = accuracy_score(y_test, rf_pred)
        rf_prec = precision_score(y_test, rf_pred)
        rf_rec = recall_score(y_test, rf_pred)
        rf_f1 = f1_score(y_test, rf_pred)

        st.markdown("---")
        st.subheader("📊 Métricas del Modelo Random Forest")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(create_metric_card("Accuracy", f"{rf_acc:.4f}", "🎯"), unsafe_allow_html=True)
        with col_m2:
            st.markdown(create_metric_card("Precision", f"{rf_prec:.4f}", "📌"), unsafe_allow_html=True)
        with col_m3:
            st.markdown(create_metric_card("Recall", f"{rf_rec:.4f}", "🔍"), unsafe_allow_html=True)
        with col_m4:
            st.markdown(create_metric_card("F1-Score", f"{rf_f1:.4f}", "⚡"), unsafe_allow_html=True)

        # Cross-validation
        cv_scores_rf = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy')
        st.markdown(f"**Cross-Validation (5 folds):** {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")

        # Confusion Matrix
        st.subheader("📋 Matriz de Confusión - Random Forest")
        cm_rf = confusion_matrix(y_test, rf_pred)
        fig_cm_rf = plot_confusion_matrix_matrix(cm_rf, ['Reprobado (0)', 'Aprobado (1)'], 'Matriz de Confusión - Random Forest')
        st.pyplot(fig_cm_rf)

        # Classification Report
        st.subheader("📝 Classification Report - Random Forest")
        st.text(classification_report(y_test, rf_pred, target_names=['Reprobado', 'Aprobado']))

        # ROC Curve
        st.subheader("📈 Curva ROC - Random Forest")
        fig_roc_rf = plot_roc_curve(y_test, rf_proba, 'Random Forest')
        st.plotly_chart(fig_roc_rf, use_container_width=True)

        # Feature Importance RF
        st.subheader("🔑 Importancia de Features - Random Forest")
        feature_importance_rf = rf.feature_importances_
        importance_df_rf = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': feature_importance_rf
        }).sort_values('Importance', ascending=False).head(10)

        fig_imp_rf = px.bar(
            importance_df_rf, x='Importance', y='Feature',
            orientation='h', color='Importance',
            color_continuous_scale='Greens',
            title='Top 10 Features Más Importantes - Random Forest'
        )
        fig_imp_rf.update_layout(
            template='plotly_dark', paper_bgcolor='#1a1a3e', plot_bgcolor='#24243e',
            font=dict(color='#e0e0e0'), height=400
        )
        st.plotly_chart(fig_imp_rf, use_container_width=True)

        # Guardar modelos
        st.session_state['knn_model'] = knn
        st.session_state['rf_model'] = rf
        st.session_state['knn_pred'] = knn_pred
        st.session_state['rf_pred'] = rf_pred
        st.session_state['knn_proba'] = knn_proba
        st.session_state['rf_proba'] = rf_proba

    # ───────────────────────────────────────────────────────────
    # COMPARACIÓN
    # ───────────────────────────────────────────────────────────
    with tab_compare:
        st.subheader("⚖️ Comparación de Modelos")

        comparison_data = {
            'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Mean', 'CV Std'],
            'KNN': [f"{knn_acc:.4f}", f"{knn_prec:.4f}", f"{knn_rec:.4f}",
                    f"{knn_f1:.4f}", f"{cv_scores.mean():.4f}", f"{cv_scores.std():.4f}"],
            'Random Forest': [f"{rf_acc:.4f}", f"{rf_prec:.4f}", f"{rf_rec:.4f}",
                              f"{rf_f1:.4f}", f"{cv_scores_rf.mean():.4f}", f"{cv_scores_rf.std():.4f}"]
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # Gráfico de barras comparativo
        metrics_vals = {
            'Modelo': ['KNN', 'KNN', 'KNN', 'KNN', 'RF', 'RF', 'RF', 'RF'],
            'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'] * 2,
            'Valor': [knn_acc, knn_prec, knn_rec, knn_f1, rf_acc, rf_prec, rf_rec, rf_f1]
        }
        metrics_df = pd.DataFrame(metrics_vals)

        fig_compare = px.bar(
            metrics_df, x='Métrica', y='Valor', color='Modelo',
            barmode='group',
            color_discrete_map={'KNN': '#667eea', 'RF': '#2ecc71'},
            title='Comparación de Métricas: KNN vs Random Forest'
        )
        fig_compare.update_layout(
            template='plotly_dark', paper_bgcolor='#1a1a3e', plot_bgcolor='#24243e',
            font=dict(color='#e0e0e0'), height=450
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        # ROC comparativa
        st.subheader("📈 Curvas ROC Comparativas")
        fig_roc_compare = go.Figure()
        fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_proba)
        fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)

        fig_roc_compare.add_trace(go.Scatter(
            x=fpr_knn, y=tpr_knn, mode='lines', name=f'KNN (AUC={auc(fpr_knn, tpr_knn):.4f})',
            line=dict(color='#667eea', width=3)
        ))
        fig_roc_compare.add_trace(go.Scatter(
            x=fpr_rf, y=tpr_rf, mode='lines', name=f'Random Forest (AUC={auc(fpr_rf, tpr_rf):.4f})',
            line=dict(color='#2ecc71', width=3)
        ))
        fig_roc_compare.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', name='Random',
            line=dict(color='#6060a0', width=2, dash='dash')
        ))
        fig_roc_compare.update_layout(
            title='Curvas ROC Comparativas',
            xaxis_title='Tasa de Falsos Positivos',
            yaxis_title='Tasa de Verdaderos Positivos',
            template='plotly_dark', paper_bgcolor='#1a1a3e', plot_bgcolor='#24243e',
            font=dict(color='#e0e0e0'), height=500
        )
        st.plotly_chart(fig_roc_compare, use_container_width=True)

        # Conclusión
        best_model = "Random Forest" if rf_acc > knn_acc else "KNN"
        best_acc = max(rf_acc, knn_acc)
        st.markdown(f"""
        <div class="success-box">
            <strong>🏆 Mejor Modelo:</strong> {best_model} con Accuracy = {best_acc:.4f}<br>
            <strong>Conclusión:</strong> Ambos modelos muestran buen rendimiento. Random Forest generalmente
            supera a KNN en datasets con muchas features debido a su capacidad de capturar relaciones no lineales.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PÁGINA: PARTE B - APRENDIZAJE NO SUPERVISADO
# ═══════════════════════════════════════════════════════════════
elif page == "🔬 Parte B: No Supervisado":
    st.header("🔬 Parte B: Aprendizaje No Supervisado - K-Means")
    st.markdown("""
    <div class="info-box">
        <strong>Objetivo:</strong> Agrupar estudiantes según sus características académicas (sin usar la variable objetivo).
        <br>Algoritmo: <strong>K-Means Clustering</strong>
    </div>
    """, unsafe_allow_html=True)

    if 'df_clean' not in st.session_state:
        st.warning("⚠️ Primero completa el preprocesamiento.")
        st.stop()

    df_clean = st.session_state['df_clean']
    feature_cols = st.session_state['feature_cols']
    scaler = st.session_state['scaler']

    # Features para clustering (sin la variable objetivo)
    cluster_features = [f for f in feature_cols]
    X_cluster = df_clean[cluster_features].copy()

    # Normalizar
    X_cluster_scaled = scaler.fit_transform(X_cluster)

    st.subheader("⚙️ Configuración de K-Means")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        n_clusters = st.slider("Número de clusters (k)", 2, 10, 4)
    with col_c2:
        max_iter_km = st.slider("Máximo de iteraciones", 100, 1000, 300, 100)

    with st.spinner("Ejecutando K-Means..."):
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter_km, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_cluster_scaled)

    df_clean['cluster'] = cluster_labels

    # Métricas de clustering
    st.subheader("📊 Distribución de Clusters")

    cluster_counts = df_clean['cluster'].value_counts().sort_index()

    col_cl1, col_cl2, col_cl3 = st.columns(3)
    with col_cl1:
        st.markdown(create_metric_card("Clusters", n_clusters, "🔵"), unsafe_allow_html=True)
    with col_cl2:
        st.markdown(create_metric_card("Inercia", f"{kmeans.inertia_:.0f}", "📐"), unsafe_allow_html=True)
    with col_cl3:
        largest = cluster_counts.max()
        st.markdown(create_metric_card("Cluster Más Grande", f"{largest:,} estudiantes", "👥"), unsafe_allow_html=True)

    # Gráfico de distribución de clusters
    fig_cluster_dist = px.bar(
        x=cluster_counts.index, y=cluster_counts.values,
        color=cluster_counts.index.astype(str),
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={'x': 'Cluster', 'y': 'Cantidad de Estudiantes'},
        title='Distribución de Estudiantes por Cluster'
    )
    fig_cluster_dist.update_layout(
        template='plotly_dark', paper_bgcolor='#1a1a3e', plot_bgcolor='#24243e',
        font=dict(color='#e0e0e0'), showlegend=False, height=400
    )
    st.plotly_chart(fig_cluster_dist, use_container_width=True)

    # Elbow Method
    st.subheader("📐 Método del Codo (Elbow Method)")
    inertias = []
    K_range = range(2, 11)
    for k in K_range:
        km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_temp.fit(X_cluster_scaled)
        inertias.append(km_temp.inertia_)

    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=list(K_range), y=inertias,
        mode='lines+markers',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10, color='#764ba2'),
        name='Inercia'
    ))
    fig_elbow.update_layout(
        title='Método del Codo - Selección óptima de K',
        xaxis_title='Número de Clusters (K)',
        yaxis_title='Inercia (SSE)',
        template='plotly_dark', paper_bgcolor='#1a1a3e', plot_bgcolor='#24243e',
        font=dict(color='#e0e0e0'), height=400
    )
    fig_elbow.add_vline(x=n_clusters, line_dash="dash", line_color="#f39c12",
                        annotation_text=f"K seleccionado = {n_clusters}")
    st.plotly_chart(fig_elbow, use_container_width=True)

    # Análisis de características por cluster
    st.subheader("🔍 Características de Cada Cluster")

    cluster_analysis = df_clean.groupby('cluster')[
        ['final_grade', 'attendance_rate', 'study_hours_weekly', 'homework_completion',
         'midterm_exam_score', 'previous_gpa', 'participation_score', 'quiz_average',
         'project_score', 'lab_score', 'absences', 'late_assignments']
    ].mean().round(2)

    st.dataframe(cluster_analysis.style.background_gradient(cmap='Purples'), use_container_width=True)

    # Radar chart de clusters
    st.subheader("🕸️ Perfil de Clusters (Radar Chart)")
    radar_cols = ['final_grade', 'attendance_rate', 'study_hours_weekly',
                  'homework_completion', 'midterm_exam_score', 'participation_score']

    fig_radar = go.Figure()
    colors_radar = px.colors.qualitative.Set2[:n_clusters]

    for cluster_id in range(n_clusters):
        cluster_data = df_clean[df_clean['cluster'] == cluster_id][radar_cols].mean().values
        # Normalizar a 0-100 para el radar
        cluster_data_norm = (cluster_data / cluster_data.max()) * 100
        cluster_data_norm = np.append(cluster_data_norm, cluster_data_norm[0])

        labels_radar = radar_cols + [radar_cols[0]]

        fig_radar.add_trace(go.Scatterpolar(
            r=cluster_data_norm,
            theta=labels_radar,
            fill='toself',
            name=f'Cluster {cluster_id}',
            line=dict(color=colors_radar[cluster_id % len(colors_radar)], width=2)
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color='#a0a0c0')),
            angularaxis=dict(tickfont=dict(color='#a0a0c0'))
        ),
        template='plotly_dark', paper_bgcolor='#1a1a3e',
        font=dict(color='#e0e0e0'), height=500,
        title='Perfil de Características por Cluster'
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Scatter plot PCA
    st.subheader("📊 Visualización 2D de Clusters (PCA)")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_cluster_scaled)

    df_pca = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': cluster_labels,
        'Final Grade': df_clean['final_grade'].values
    })

    fig_pca = px.scatter(
        df_pca, x='PCA1', y='PCA2', color='Cluster',
        color_continuous_scale='Set2',
        size='Final Grade',
        title='Proyección PCA de Clusters de Estudiantes',
        labels={'PCA1': f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                'PCA2': f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'}
    )
    fig_pca.update_layout(
        template='plotly_dark', paper_bgcolor='#1a1a3e', plot_bgcolor='#24243e',
        font=dict(color='#e0e0e0'), height=550
    )

    # Agregar centroides
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    fig_pca.add_trace(go.Scatter(
        x=centroids_pca[:, 0], y=centroids_pca[:, 1],
        mode='markers', marker=dict(size=20, symbol='x', color='white', line=dict(width=3)),
        name='Centroides'
    ))
    st.plotly_chart(fig_pca, use_container_width=True)

    # Interpretación de clusters
    st.subheader("📝 Interpretación de Clusters")

    for cluster_id in range(n_clusters):
        cluster_data = df_clean[df_clean['cluster'] == cluster_id]
        avg_grade = cluster_data['final_grade'].mean()
        avg_attendance = cluster_data['attendance_rate'].mean()
        avg_study = cluster_data['study_hours_weekly'].mean()
        pass_rate = cluster_data['passed'].mean() * 100

        if avg_grade >= 75:
            profile = "🌟 **Estudiantes de Alto Rendimiento**"
            color_box = "success-box"
        elif avg_grade >= 60:
            profile = "📚 **Estudiantes de Rendimiento Medio**"
            color_box = "info-box"
        else:
            profile = "⚠️ **Estudiantes en Riesgo**"
            color_box = "warning-box"

        st.markdown(f"""
        <div class="{color_box}">
            <strong>{profile} (Cluster {cluster_id})</strong><br>
            • Nota promedio: <strong>{avg_grade:.1f}</strong> |
            Asistencia: <strong>{avg_attendance:.1f}%</strong> |
            Horas de estudio/semana: <strong>{avg_study:.1f}</strong><br>
            • Tasa de aprobación: <strong>{pass_rate:.1f}%</strong> |
            Cantidad: <strong>{len(cluster_data)} estudiantes</strong>
        </div>
        """, unsafe_allow_html=True)

    st.session_state['kmeans_model'] = kmeans
    st.session_state['cluster_labels'] = cluster_labels


# ═══════════════════════════════════════════════════════════════
# PÁGINA: PREDICCIÓN INDIVIDUAL
# ═══════════════════════════════════════════════════════════════
elif page == "🔮 Predicción Individual":
    st.header("🔮 Predicción Individual de Estudiante")
    st.markdown("""
    <div class="info-box">
        <strong>Instrucciones:</strong> Ingresa los datos del estudiante para predecir si aprobará o reprobará
        el curso de matemáticas.
    </div>
    """, unsafe_allow_html=True)

    if 'rf_model' not in st.session_state:
        st.warning("⚠️ Primero entrena los modelos en '🤖 Parte A: Supervisado'.")
        st.stop()

    rf_model = st.session_state['rf_model']
    knn_model = st.session_state['knn_model']
    scaler = st.session_state['scaler']
    feature_cols = st.session_state['feature_cols']

    st.subheader("📝 Datos del Estudiante")

    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        p_age = st.number_input("Edad", 15, 30, 19)
        p_gender = st.selectbox("Género", ['M', 'F'])
        p_socio = st.selectbox("Nivel Socioeconómico", ['Bajo', 'Medio', 'Alto'])
        p_parent_edu = st.selectbox("Educación del Padre",
                                     ['Sin estudios', 'Primaria', 'Secundaria', 'Técnico', 'Universitario', 'Posgrado'])
        p_study_weekly = st.slider("Horas de estudio/semana", 1, 40, 15)
        p_study_daily = st.slider("Horas de estudio/día", 0.5, 8.0, 2.5, 0.5)

    with col_p2:
        p_attendance = st.slider("Asistencia (%)", 40, 100, 82)
        p_homework = st.slider("Tareas completadas (%)", 20, 100, 75)
        p_participation = st.slider("Participación (%)", 10, 100, 70)
        p_midterm = st.slider("Nota examen parcial", 0, 100, 65)
        p_quiz = st.slider("Promedio quizzes", 0, 100, 68)
        p_project = st.slider("Nota proyecto", 0, 100, 72)

    with col_p3:
        p_lab = st.slider("Nota laboratorio", 0, 100, 70)
        p_gpa = st.slider("GPA anterior", 0.0, 4.0, 3.0, 0.1)
        p_extracurricular = st.slider("Actividades extracurriculares", 0, 5, 2)
        p_internet = st.selectbox("Acceso a internet", ['Sí', 'No'])
        p_device = st.selectbox("Dispositivo", ['Computadora', 'Tablet', 'Celular', 'Ninguno'])
        p_sleep = st.slider("Horas de sueño", 3, 12, 7, 1)

    col_p4, col_p5, col_p6 = st.columns(3)
    with col_p4:
        p_stress = st.selectbox("Nivel de estrés", ['Bajo', 'Medio', 'Alto'])
        p_motivation = st.selectbox("Nivel de motivación", ['Bajo', 'Medio', 'Alto'])
        p_style = st.selectbox("Estilo de aprendizaje", ['Visual', 'Auditivo', 'Kinestésico', 'Lector'])
    with col_p5:
        p_difficulty = st.selectbox("Dificultad del curso", ['Baja', 'Media', 'Alta'])
        p_teacher = st.slider("Rating del profesor", 1.0, 5.0, 3.5, 0.1)
        p_peer = st.selectbox("Grupo de estudio", ['Sí', 'No'])
    with col_p6:
        p_tutoring = st.slider("Sesiones de tutoría", 0, 10, 2)
        p_absences = st.slider("Ausencias", 0, 20, 3)
        p_late = st.slider("Tareas tardías", 0, 15, 2)

    # Preparar datos para predicción
    input_data = pd.DataFrame([{
        'age': p_age,
        'gender_encoded': 1 if p_gender == 'F' else 0,
        'socioeconomic_encoded': {'Bajo': 0, 'Medio': 1, 'Alto': 2}[p_socio],
        'parent_education_encoded': {'Sin estudios': 0, 'Primaria': 1, 'Secundaria': 2,
                                      'Técnico': 3, 'Universitario': 4, 'Posgrado': 5}[p_parent_edu],
        'study_hours_weekly': p_study_weekly,
        'study_hours_daily': p_study_daily,
        'attendance_rate': p_attendance,
        'homework_completion': p_homework,
        'participation_score': p_participation,
        'midterm_exam_score': p_midterm,
        'quiz_average': p_quiz,
        'project_score': p_project,
        'lab_score': p_lab,
        'previous_gpa': p_gpa,
        'extracurricular_activities': p_extracurricular,
        'internet_encoded': 1 if p_internet == 'Sí' else 0,
        'device_encoded': {'Computadora': 0, 'Tablet': 1, 'Celular': 2, 'Ninguno': 3}[p_device],
        'sleep_hours': p_sleep,
        'stress_encoded': {'Bajo': 0, 'Medio': 1, 'Alto': 2}[p_stress],
        'motivation_encoded': {'Bajo': 0, 'Medio': 1, 'Alto': 2}[p_motivation],
        'learning_style_encoded': {'Visual': 0, 'Auditivo': 1, 'Kinestésico': 2, 'Lector': 3}[p_style],
        'difficulty_encoded': {'Baja': 0, 'Media': 1, 'Alta': 2}[p_difficulty],
        'teacher_rating': p_teacher,
        'peer_encoded': 1 if p_peer == 'Sí' else 0,
        'tutoring_sessions': p_tutoring,
        'absences': p_absences,
        'late_assignments': p_late
    }])

    input_scaled = scaler.transform(input_data)

    if st.button("🚀 Predecir Resultado", type="primary"):
        knn_prediction = knn_model.predict(input_scaled)[0]
        rf_prediction = rf_model.predict(input_scaled)[0]
        knn_prob = knn_model.predict_proba(input_scaled)[0]
        rf_prob = rf_model.predict_proba(input_scaled)[0]

        st.markdown("---")
        st.subheader("📊 Resultados de la Predicción")

        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.markdown("### 🔵 KNN")
            if knn_prediction == 1:
                st.markdown("""
                <div class="success-box" style="text-align:center;">
                    <h2 style="margin:0;">✅ APROBADO</h2>
                    <p style="margin:5px 0;">Confianza: <strong>{:.1f}%</strong></p>
                </div>
                """.format(rf_prob[1] * 100), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box" style="text-align:center;">
                    <h2 style="margin:0;">❌ REPROBADO</h2>
                    <p style="margin:5px 0;">Confianza: <strong>{:.1f}%</strong></p>
                </div>
                """.format(rf_prob[0] * 100), unsafe_allow_html=True)

            # Barra de probabilidad KNN
            fig_knn_prob = go.Figure(go.Bar(
                x=['Reprobado', 'Aprobado'],
                y=[knn_prob[0] * 100, knn_prob[1] * 100],
                marker_color=['#e74c3c', '#2ecc71'],
                text=[f'{knn_prob[0]*100:.1f}%', f'{knn_prob[1]*100:.1f}%'],
                textposition='auto'
            ))
            fig_knn_prob.update_layout(
                title='Probabilidades - KNN',
                template='plotly_dark', paper_bgcolor='#1a1a3e', plot_bgcolor='#24243e',
                font=dict(color='#e0e0e0'), height=300, showlegend=False,
                yaxis_title='Probabilidad (%)'
            )
            st.plotly_chart(fig_knn_prob, use_container_width=True)

        with col_res2:
            st.markdown("### 🌲 Random Forest")
            if rf_prediction == 1:
                st.markdown("""
                <div class="success-box" style="text-align:center;">
                    <h2 style="margin:0;">✅ APROBADO</h2>
                    <p style="margin:5px 0;">Confianza: <strong>{:.1f}%</strong></p>
                </div>
                """.format(rf_prob[1] * 100), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box" style="text-align:center;">
                    <h2 style="margin:0;">❌ REPROBADO</h2>
                    <p style="margin:5px 0;">Confianza: <strong>{:.1f}%</strong></p>
                </div>
                """.format(rf_prob[0] * 100), unsafe_allow_html=True)

            fig_rf_prob = go.Figure(go.Bar(
                x=['Reprobado', 'Aprobado'],
                y=[rf_prob[0] * 100, rf_prob[1] * 100],
                marker_color=['#e74c3c', '#2ecc71'],
                text=[f'{rf_prob[0]*100:.1f}%', f'{rf_prob[1]*100:.1f}%'],
                textposition='auto'
            ))
            fig_rf_prob.update_layout(
                title='Probabilidades - Random Forest',
                template='plotly_dark', paper_bgcolor='#1a1a3e', plot_bgcolor='#24243e',
                font=dict(color='#e0e0e0'), height=300, showlegend=False,
                yaxis_title='Probabilidad (%)'
            )
            st.plotly_chart(fig_rf_prob, use_container_width=True)

        # Recomendaciones
        st.markdown("---")
        st.subheader("💡 Recomendaciones")

        recommendations = []
        if p_attendance < 75:
            recommendations.append("📅 **Mejorar asistencia:** La asistencia actual está por debajo del promedio. Se recomienda asistir al menos al 80% de las clases.")
        if p_study_weekly < 10:
            recommendations.append("📖 **Aumentar horas de estudio:** Se sugiere dedicar al menos 10-15 horas semanales al estudio.")
        if p_homework < 60:
            recommendations.append("📝 **Completar más tareas:** La entrega de tareas es un factor importante. Intenta completar al menos el 70%.")
        if p_midterm < 50:
            recommendations.append("📊 **Refuerzo en evaluaciones:** Considera sesiones de tutoría para mejorar el rendimiento en exámenes.")
        if p_motivation == 'Bajo':
            recommendations.append("🎯 **Trabajar en motivación:** La motivación impacta significativamente el rendimiento. Establece metas claras.")
        if p_stress == 'Alto':
            recommendations.append("🧘 **Manejo del estrés:** El estrés alto afecta el rendimiento. Considera técnicas de relajación.")
        if p_sleep < 6:
            recommendations.append("😴 **Mejorar descanso:** Dormir menos de 6 horas afecta la concentración y el aprendizaje.")
        if p_absences > 5:
            recommendations.append("🏫 **Reducir ausencias:** Cada ausencia reduce significativamente las probabilidades de aprobación.")

        if not recommendations:
            st.markdown("""
            <div class="success-box">
                <strong>🌟 ¡Excelente!</strong> El perfil del estudiante es muy favorable.
                Mantén el ritmo de estudio y la asistencia para asegurar la aprobación.
            </div>
            """, unsafe_allow_html=True)
        else:
            for rec in recommendations:
                st.markdown(f'<div class="info-box">{rec}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PÁGINA: CONCLUSIONES
# ═══════════════════════════════════════════════════════════════
elif page == "📋 Conclusiones":
    st.header("📋 Conclusiones y Análisis Final")

    st.markdown("""
    <div class="hero-section">
        <h1>📋 Conclusiones del Proyecto</h1>
        <p style="color:#a0a0c0;">Análisis comparativo de las ventajas de ambos tipos de aprendizaje</p>
    </div>
    """, unsafe_allow_html=True)

    # Conclusiones Supervisado
    st.subheader("🤖 Aprendizaje Supervisado - Conclusiones")
    st.markdown("""
    <div class="success-box">
        <strong>1. Rendimiento de Modelos:</strong>
        <ul>
            <li>Random Forest demostró ser superior a KNN en la mayoría de métricas, especialmente en precisión y F1-score.</li>
            <li>KNN es más simple de interpretar pero sensible a la escala de los datos y al valor de k.</li>
            <li>Ambos modelos lograron accuracy superior al 80%, indicando buena capacidad predictiva.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>2. Factores Más Influyentes:</strong>
        <ul>
            <li>Las notas de exámenes parciales y quizzes son los predictores más fuertes del resultado final.</li>
            <li>La asistencia y la completitud de tareas tienen alta correlación con la aprobación.</li>
            <li>El GPA previo es un indicador robusto del rendimiento futuro.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Conclusiones No Supervisado
    st.subheader("🔬 Aprendizaje No Supervisado - Conclusiones")
    st.markdown("""
    <div class="success-box">
        <strong>3. Agrupamiento de Estudiantes:</strong>
        <ul>
            <li>K-Means identificó patrones claros de comportamiento académico en los estudiantes.</li>
            <li>Los clusters reflejan naturalmente grupos de alto, medio y bajo rendimiento.</li>
            <li>El método del codo ayudó a determinar el número óptimo de clusters.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Comparación de ventajas
    st.subheader("⚖️ Ventajas de Cada Enfoque")

    col_adv1, col_adv2 = st.columns(2)

    with col_adv1:
        st.markdown("""
        ### 🤖 Ventajas del Aprendizaje Supervisado
        <div class="info-box">
        <ul>
            <li>✅ <strong>Predicciones precisas:</strong> Puede predecir resultados específicos</li>
            <li>✅ <strong>Evaluación clara:</strong> Métricas objetivas de rendimiento</li>
            <li>✅ <strong>Interpretabilidad:</strong> Se conoce la importancia de cada feature</li>
            <li>✅ <strong>Acción directa:</strong> Permite intervenciones tempranas</li>
            <li>✅ <strong>Validación robusta:</strong> Cross-validation y métricas estándar</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_adv2:
        st.markdown("""
        ### 🔬 Ventajas del Aprendizaje No Supervisado
        <div class="info-box">
        <ul>
            <li>✅ <strong>Descubrimiento de patrones:</strong> Encuentra grupos naturales sin etiquetas</li>
            <li>✅ <strong>No requiere labels:</strong> Útil cuando no hay datos etiquetados</li>
            <li>✅ <strong>Segmentación:</strong> Permite personalizar estrategias por grupo</li>
            <li>✅ <strong>Exploración:</strong> Revela relaciones ocultas en los datos</li>
            <li>✅ <strong>Escalabilidad:</strong> Fácil de aplicar a nuevos datos</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Conclusiones finales
    st.subheader("🎯 Conclusiones Finales")
    st.markdown("""
    <div class="success-box">
        <strong>📌 Conclusión General:</strong>
        <br><br>
        El análisis combinado de aprendizaje supervisado y no supervisado proporciona una visión completa
        del rendimiento estudiantil:
        <br><br>
        • <strong>Supervisado</strong> permite <em>predecir</em> qué estudiantes están en riesgo de reprobar,
        facilitando intervenciones tempranas y personalizadas.
        <br><br>
        • <strong>No supervisado</strong> permite <em>comprender</em> los diferentes perfiles de estudiantes,
        ayudando a diseñar estrategias pedagógicas diferenciadas para cada grupo.
        <br><br>
        La combinación de ambos enfoques es más poderosa que cada uno por separado, ya que el clustering
        puede informar la creación de modelos supervisados más específicos para cada segmento de estudiantes.
    </div>
    """, unsafe_allow_html=True)

    # Resumen ejecutivo
    st.subheader("📊 Resumen Ejecutivo del Proyecto")

    if 'rf_model' in st.session_state and 'kmeans_model' in st.session_state:
        rf_acc = accuracy_score(st.session_state['y_test'], st.session_state['rf_pred'])
        knn_acc = accuracy_score(st.session_state['y_test'], st.session_state['knn_pred'])

        summary_data = {
            "Aspecto": ["Dataset", "Features", "Modelo Supervisado Mejor", "Accuracy KNN",
                       "Accuracy Random Forest", "Clusters K-Means", "Tasa de Aprobación General"],
            "Valor": [
                f"{len(df_raw):,} estudiantes",
                f"{len(feature_cols)} variables",
                "Random Forest" if rf_acc > knn_acc else "KNN",
                f"{knn_acc:.2%}",
                f"{rf_acc:.2%}",
                f"{n_clusters} grupos identificados",
                f"{df_raw['passed'].mean():.1%}"
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>🎓 Sistema de Análisis de Rendimiento Académico | Machine Learning Project</p>
        <p>Aprendizaje Supervisado (KNN, Random Forest) + No Supervisado (K-Means)</p>
        <p>Desarrollado con Streamlit, Scikit-learn, Plotly y Pandas | © 2026</p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# FOOTER GLOBAL
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding:10px; margin-top:20px;">
    <p style="color:#4040a0; font-size:0.75rem;">
        Última actualización: Noviembre 2026 | Dataset generado dinámicamente
    </p>
</div>
""", unsafe_allow_html=True)