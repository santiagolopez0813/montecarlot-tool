import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Simulador Monte Carlo", layout="wide")
st.title("📈 Simulador de Escenarios Financieros (Monte Carlo)")

# 1. DICCIONARIOS DE ACTIVOS PREDEFINIDOS
diccionario_activos = {
    "Acción": ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "NFLX", "COIN", "Otro (Escribir manual)"],
    "ETF": ["SPY", "QQQ", "VOO", "DIA", "IWM", "VTI", "KORU", "Otro (Escribir manual)"],
    "Criptomoneda": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "Otro (Escribir manual)"]
}

# Barra lateral para los controles
st.sidebar.header("Configuración de Parámetros")
tipo_activo = st.sidebar.selectbox("Tipo de Activo", ["Acción", "ETF", "Criptomoneda"])

# 2. LÓGICA DE BÚSQUEDA DINÁMICA
seleccion = st.sidebar.selectbox(
    "Selecciona el Ticker rápido", 
    options=diccionario_activos[tipo_activo]
)

# Si el usuario selecciona "Otro", mostramos una caja de texto para que escriba lo que quiera
if seleccion == "Otro (Escribir manual)":
    ticker = st.sidebar.text_input("Escribe el Ticker exacto (Ej: GME, MSFT, HAPI)").upper()
else:
    ticker = seleccion

dias = st.sidebar.slider("Días a predecir", min_value=10, max_value=365, value=120)
simulaciones = st.sidebar.slider("Número de simulaciones", min_value=50, max_value=500, value=100)

if st.sidebar.button("Ejecutar Simulación"):
    # Asegurarnos de que el usuario haya escrito algo si eligió "Otro"
    if not ticker:
        st.sidebar.error("Por favor, escribe un Ticker válido.")
    else:
        with st.spinner("Descargando datos y calculando trayectorias..."):
            try:
                activo = yf.Ticker(ticker)
                data = activo.history(period="1y")
                
                if data.empty:
                    st.error(f"No se encontraron datos en Yahoo Finance para el ticker '{ticker}'. Asegúrate de que el símbolo sea correcto (ej. añade '-USD' para criptos).")
                else:
                    precios_cierre = data['Close']
                    precio_actual = float(precios_cierre.iloc[-1])
                    
                    # Calcular rendimientos y volatilidad histórica
                    rendimientos_diarios = np.log(1 + precios_cierre.pct_change().dropna())
                    drift = rendimientos_diarios.mean() - (0.5 * rendimientos_diarios.var())
                    volatilidad = rendimientos_diarios.std()

                    drift_val = drift.values[0] if hasattr(drift, 'values') else drift
                    vol_val = volatilidad.values[0] if hasattr(volatilidad, 'values') else volatilidad

                    # Generar el modelo Monte Carlo
                    Z = np.random.normal(size=(dias, simulaciones))
                    predicciones_diarias = np.exp(drift_val + vol_val * Z)
                    
                    trayectorias = np.zeros_like(predicciones_diarias)
                    trayectorias[0] = precio_actual
                    
                    for t in range(1, dias):
                        trayectorias[t] = trayectorias[t - 1] * predicciones_diarias[t]

                    # Visualización
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(trayectorias, linewidth=0.5, alpha=0.6)
                    ax.set_title(f"Simulación para {ticker} ({simulaciones} escenarios)")
                    ax.set_xlabel("Días")
                    ax.set_ylabel("Precio ($)")
                    
                    # Mejorar el formato de los números en el eje Y de la gráfica
                    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
                    
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                    # PANEL DE RESULTADOS CON FORMATO MEJORADO
                    precios_finales = trayectorias[-1]
                    precio_promedio = np.mean(precios_finales)
                    p5 = np.percentile(precios_finales, 5)
                    p95 = np.percentile(precios_finales, 95)
                    
                    st.divider()
                    st.subheader("📊 Resultados Proyectados")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Precio Actual", f"${precio_actual:,.2f}")
                    col2.metric("Precio Esperado (Promedio)", f"${precio_promedio:,.2f}")
                    col3.metric("Escenario Pesimista (P5)", f"${p5:,.2f}")
                    col4.metric("Escenario Optimista (P95)", f"${p95:,.2f}")

            except Exception as e:
                st.error(f"Error técnico detallado: {e}")
