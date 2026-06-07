import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Simulador Monte Carlo", layout="wide")
st.title("📈 Simulador de Escenarios Financieros (Monte Carlo)")

# Barra lateral para los controles
st.sidebar.header("Configuración de Parámetros")
tipo_activo = st.sidebar.selectbox("Tipo de Activo", ["Acción", "ETF", "Criptomoneda"])
# Forzamos a que el ticker esté en mayúsculas
ticker = st.sidebar.text_input("Ticker (Ej: AAPL, SPY, BTC-USD)", value="AAPL").upper()
dias = st.sidebar.slider("Días a predecir", min_value=10, max_value=365, value=120)
simulaciones = st.sidebar.slider("Número de simulaciones", min_value=50, max_value=500, value=100)

if st.sidebar.button("Ejecutar Simulación"):
    with st.spinner("Descargando datos y calculando trayectorias..."):
        try:
            # MÉTODO ACTUALIZADO MÁS ESTABLE
            activo = yf.Ticker(ticker)
            data = activo.history(period="1y")
            
            # Validación por si Yahoo Finance no devuelve datos
            if data.empty:
                st.error(f"No se encontraron datos en Yahoo Finance para el ticker '{ticker}'. Verifica el símbolo.")
            else:
                precios_cierre = data['Close']
                precio_actual = float(precios_cierre.iloc[-1])
                
                # Calcular rendimientos y volatilidad histórica
                rendimientos_diarios = np.log(1 + precios_cierre.pct_change().dropna())
                drift = rendimientos_diarios.mean() - (0.5 * rendimientos_diarios.var())
                volatilidad = rendimientos_diarios.std()

                # Extraer valores numéricos de forma segura
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
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # Panel de Resultados
                precios_finales = trayectorias[-1]
                col1, col2, col3 = st.columns(3)
                col1.metric("Precio Actual", f"${precio_actual:.2f}")
                col2.metric("Precio Esperado (Promedio)", f"${np.mean(precios_finales):.2f}")
                col3.metric("Escenario Pesimista (P5)", f"${np.percentile(precios_finales, 5):.2f}")

        except Exception as e:
            # SI FALLA, AHORA MOSTRARÁ EL ERROR REAL
            st.error(f"Error técnico detallado: {e}")
