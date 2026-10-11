import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

# 1. Configuración de página
st.set_page_config(
    page_title="Sistema EcoTránsito CDMX",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Sistema EcoTránsito CDMX")
st.markdown("Plataforma Inteligente de Diagnóstico y Monitoreo de Calidad del Aire y Flujo Vehicular (SEDEMA / SEMOVI)")

@st.cache_data
def cargar_y_limpiar_datos():
    data = {
        'ID_Registro': ['REG-001', 'REG-002', 'REG-003', 'REG-004', 'REG-005', 
                        'REG-006', 'REG-007', 'REG-008', 'REG-009', 'REG-010', 'REG-010'],
        'Timestamp': ['2026-10-01 08:00:00', '2026-10-01 09:00:00', '2026-10-01 10:00:00',
                      '2026-10-01 11:00:00', '2026-10-01 12:00:00', '2026-10-01 13:00:00',
                      '2026-10-01 14:00:00', '2026-10-01 15:00:00', '2026-10-01 16:00:00',
                      '2026-10-01 17:00:00', '2026-10-01 17:00:00'],
        'Zona_Estacion': ['Centro_Historico', 'centro_historico', 'CENTRO_HISTORICO',
                          'Zona_Oriente', 'Zona_Oriente', 'zona_oriente', 'Zona_Sur',
                          'Zona_Sur', 'Zona_Sur', 'Zona_Sur', 'Zona_Sur'],
        'Concentracion_PM25': [35.4, 42.1, -999.0, 120.5, np.nan, 310.0, 18.2, 20.1, 19.5, 19.5, 19.5],
        'Nivel_NO2': [42.1, 48.5, 51.0, 89.2, 82.0, 195.0, 22.4, -999.0, 25.1, 25.1, 25.1],
        'Indice_Trafico_Vehicular': [85.0, 92.5, 88.0, 98.0, 91.2, 15.0, 45.0, 42.1, np.nan, 41.0, 41.0],
        'Estado_Sensor': ['Normal', 'Normal', 'Alerta', 'Alerta', 'Normal', 'Falla',
                          'Normal', 'Normal', 'Normal', 'Normal', 'Normal']
    }
    df = pd.DataFrame(data)
    
    cols_num = ['Concentracion_PM25', 'Nivel_NO2', 'Indice_Trafico_Vehicular']
    df[cols_num] = df[cols_num].replace(-999.0, np.nan)
    df = df.drop_duplicates().reset_index(drop=True)
    
    df['Zona_Estacion'] = df['Zona_Estacion'].str.strip().str.replace(' ', '_').str.title()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hora_Dia'] = df['Timestamp'].dt.hour
    
    Q1 = df['Concentracion_PM25'].quantile(0.25)
    Q3 = df['Concentracion_PM25'].quantile(0.75)
    IQR = Q3 - Q1
    cond_outlier = (df['Concentracion_PM25'] < (Q1 - 1.5 * IQR)) | (df['Concentracion_PM25'] > (Q3 + 1.5 * IQR))
    df.loc[cond_outlier, 'Concentracion_PM25'] = np.nan
    
    imputer = KNNImputer(n_neighbors=2)
    df[cols_num] = imputer.fit_transform(df[cols_num])
    
    return df

df_clean = cargar_y_limpiar_datos()

# 2. Carga y filtrado de datos con barra lateral
st.sidebar.header("🔍 Filtros de Monitoreo")
zonas_disponibles = df_clean['Zona_Estacion'].unique().tolist()

zonas_seleccionadas = st.sidebar.multiselect(
    "Selecciona las Zonas de Estación:",
    options=zonas_disponibles,
    default=zonas_disponibles
)

df_filtrado = df_clean[df_clean['Zona_Estacion'].isin(zonas_seleccionadas)]

# 3. Métricas principales (KPIs)
st.subheader("📊 Indicadores Clave de Desempeño (KPIs)")
col1, col2, col3 = st.columns(3)

total_lecturas = len(df_filtrado)
promedio_pm25 = df_filtrado['Concentracion_PM25'].mean() if total_lecturas > 0 else 0.0
max_trafico = df_filtrado['Indice_Trafico_Vehicular'].max() if total_lecturas > 0 else 0.0

with col1:
    st.metric(label="Total de Lecturas Activas", value=f"{total_lecturas}")
with col2:
    st.metric(label="Promedio de PM2.5 (μg/m³)", value=f"{promedio_pm25:.2f} μg/m³")
with col3:
    st.metric(label="Índice Máximo de Tráfico Vehicular", value=f"{max_trafico:.1f}")

st.divider()

# 4. Pestañas interactivas de visualización (Plotly + Seaborn)
tab1, tab2 = st.tabs([
    "📈 Dispersión Tráfico vs PM2.5 (Plotly)", 
    "🔥 Mapa de Calor de Correlación (Seaborn)"
])

with tab1:
    st.markdown("#### Relación entre Flujo Vehicular y Concentración de Partículas $PM_{2.5}$")
    if not df_filtrado.empty:
        fig_scatter = px.scatter(
            df_filtrado,
            x="Indice_Trafico_Vehicular",
            y="Concentracion_PM25",
            color="Zona_Estacion",
            size="Nivel_NO2",
            hover_data=["ID_Registro", "Hora_Dia", "Estado_Sensor"],
            labels={
                "Indice_Trafico_Vehicular": "Índice de Tráfico Vehicular",
                "Concentracion_PM25": "Concentración PM2.5 (μg/m³)",
                "Zona_Estacion": "Zona Estación"
            },
            title="Concentración de PM2.5 según el Índice de Tráfico Vehicular por Zona",
            template="plotly_white"
        )
        fig_scatter.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Por favor selecciona al menos una zona en la barra lateral para visualizar los datos.")

with tab2:
    st.markdown("#### Matriz de Correlación de Variables Ambientales y Viales")
    if len(df_filtrado) > 1:
        cols_corr = ['Concentracion_PM25', 'Nivel_NO2', 'Indice_Trafico_Vehicular', 'Hora_Dia']
        matriz_corr = df_filtrado[cols_corr].corr(method='spearman')
        
        fig_heat, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            matriz_corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            ax=ax
        )
        ax.set_title("Mapa de Calor de Correlación (Spearman)")
        st.pyplot(fig_heat)
    else:
        st.warning("Se requieren al menos 2 registros para calcular la matriz de correlación.")

with st.expander("📋 Ver tabla de datos procesados (df_clean)"):
    st.dataframe(df_filtrado, use_container_width=True)
