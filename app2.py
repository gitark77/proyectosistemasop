import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

def initialize_session_state():
    if 'analysis_combined_df' not in st.session_state:
        st.session_state.analysis_combined_df = None
    if 'generated_fig' not in st.session_state:
        st.session_state.generated_fig = None

def analyze_data(df, analysis_type, column):
    try:
        analysis_functions = {
            'Promedio por OFICINA': lambda df, col: df.groupby('OFICINA')[col].mean().reset_index(),
            'Total por OFICINA': lambda df, col: df.groupby('OFICINA')[col].sum().reset_index(),
            'Mediana por OFICINA': lambda df, col: df.groupby('OFICINA')[col].median().reset_index(),
            'Desviación estándar por OFICINA': lambda df, col: df.groupby('OFICINA')[col].std().reset_index(),
            'Promedio por LINEA': lambda df, col: df.groupby('LINEA')[col].mean().reset_index(),
            'Total por LINEA': lambda df, col: df.groupby('LINEA')[col].sum().reset_index(),
            'Mediana por LINEA': lambda df, col: df.groupby('LINEA')[col].median().reset_index(),
            'Desviación estándar por LINEA': lambda df, col: df.groupby('LINEA')[col].std().reset_index(),
            'Promedio por GRUPO': lambda df, col: df.groupby('GRUPO')[col].mean().reset_index(),
            'Total por GRUPO': lambda df, col: df.groupby('GRUPO')[col].sum().reset_index(),
            'Máximo por OFICINA': lambda df, col: df.groupby('OFICINA')[col].max().reset_index(),
            'Mínimo por OFICINA': lambda df, col: df.groupby('OFICINA')[col].min().reset_index(),
            'Máximo por LINEA': lambda df, col: df.groupby('LINEA')[col].max().reset_index(),
            'Mínimo por LINEA': lambda df, col: df.groupby('LINEA')[col].min().reset_index(),
        }
        return analysis_functions.get(analysis_type, lambda df, col: pd.DataFrame())(df, column)
    except Exception as e:
        st.error(f"Error durante el análisis de datos: {e}")
        return pd.DataFrame()

def generate_graph(df, analysis_type, graph_type='bar'):
    try:
        fig, ax = plt.subplots()
        if graph_type == 'bar':
            df.plot(kind='bar', x=df.columns[0], y=df.columns[1], ax=ax)
        elif graph_type == 'line':
            df.plot(kind='line', x=df.columns[0], y=df.columns[1], ax=ax)
        elif graph_type == 'scatter':
            df.plot(kind='scatter', x=df.columns[0], y=df.columns[1], ax=ax)
        elif graph_type == 'pie':
            df.set_index(df.columns[0]).plot(kind='pie', y=df.columns[1], ax=ax, autopct='%1.1f%%')
        ax.set_title(analysis_type)
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[1])
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        return fig
    except Exception as e:
        st.error(f"Error al generar el gráfico: {e}")
        return None

def save_graph(fig, format='png'):
    try:
        buf = BytesIO()
        fig.savefig(buf, format=format)
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Error al guardar el gráfico: {e}")
        return None

def read_excel_sheet(uploaded_file, sheet_name):
    try:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"Error al leer la hoja {sheet_name}: {e}")
        return None

def predict_data(df, feature_column, target_column, model_type):
    try:
        X = df[[feature_column]]
        y = df[target_column]
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        if model_type == 'Linear Regression':
            model = LinearRegression()
        elif model_type == 'Random Forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'SVR':
            model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        elif model_type == 'Ridge Regression':
            model = Ridge()
        elif model_type == 'Lasso Regression':
            model = Lasso()
    
        model.fit(X_train, y_train)
    
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
    
        predictions = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
    
        return model, predictions, mse
    except Exception as e:
        st.error(f"Error durante la predicción de datos: {e}")
        return None, pd.DataFrame(), None

def main():
    initialize_session_state()
    
    st.title("Análisis de datos Básico")
    st.write("Utilice esta sección para analizar datos después de haberlos procesado y validado.")
    
    uploaded_file = st.file_uploader("Sube un archivo Excel", type=["xlsx"], key="analysis_file_uploader")
    
    if uploaded_file is not None:
        sheet_options = {
            "Datos validados": "Sheet1",
            "Datos defectuosos (valores nulos)": "Sheet2"
        }
        sheet_name_display = st.selectbox("Seleccione la hoja a analizar", list(sheet_options.keys()), key="analysis_sheet_selectbox")
        sheet_name = sheet_options[sheet_name_display]
        
        if sheet_name:
            df = read_excel_sheet(uploaded_file, sheet_name)
            if df is not None:
                st.session_state.analysis_combined_df = df
                st.write(f"Datos de la hoja '{sheet_name_display}' del archivo Excel:")
                st.dataframe(df)
                
                numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
                if not numeric_columns:
                    st.warning("No hay columnas numéricas en los datos. Por favor, seleccione un archivo con datos numéricos.")
                    return
                
                column = st.selectbox("Seleccione la columna para el análisis", numeric_columns, key="analysis_column_selectbox")
    
                analysis_type = st.selectbox("Seleccione el tipo de análisis", 
                                             ["Promedio por OFICINA", 
                                              "Total por OFICINA", 
                                              "Mediana por OFICINA",
                                              "Desviación estándar por OFICINA",
                                              "Promedio por LINEA", 
                                              "Total por LINEA",
                                              "Mediana por LINEA",
                                              "Desviación estándar por LINEA",
                                              "Promedio por GRUPO",
                                              "Total por GRUPO",
                                              "Máximo por OFICINA",
                                              "Mínimo por OFICINA",
                                              "Máximo por LINEA",
                                              "Mínimo por LINEA"], key="analysis_type_selectbox")
                
                graph_type = st.selectbox("Seleccione el tipo de gráfico", ["bar", "line", "scatter", "pie"], key="graph_type_selectbox")
                
                if st.button("Generar gráfico de datos", key="generate_graph_button"):
                    if st.session_state.analysis_combined_df is not None:
                        analysis_result = analyze_data(st.session_state.analysis_combined_df, analysis_type, column)
                        if not analysis_result.empty:
                            st.write("Resultados del análisis de datos:")
                            st.dataframe(analysis_result)
                            st.write("Gráfico generado:")
                            fig = generate_graph(analysis_result, analysis_type, graph_type)
                            
                            if fig:
                                st.session_state.generated_fig = fig 
                                st.pyplot(fig)  

                if st.session_state.generated_fig is not None:
                    buffer = save_graph(st.session_state.generated_fig, format='png')
                    if buffer:
                        filename = f"grafico_{analysis_type}.png"
                        st.download_button(
                            label="Guardar gráfico",
                            data=buffer,
                            file_name=filename,
                            mime="image/png"
                        )
                    
    st.title("Análisis de datos Avanzado - Predicciones")
    st.write("Realice predicciones basadas en los datos utilizando diferentes modelos de regresión.")
    
    if uploaded_file is not None and 'analysis_combined_df' in st.session_state:
        df = st.session_state.analysis_combined_df
        
        feature_column = st.selectbox("Seleccione la columna de características (X)", numeric_columns, key="feature_column_selectbox")
        target_column = st.selectbox("Seleccione la columna objetivo (Y)", numeric_columns, key="target_column_selectbox")
        
        model_type = st.selectbox("Seleccione el tipo de modelo", ["Linear Regression", "Random Forest", "SVR", "Ridge Regression", "Lasso Regression"], key="model_type_selectbox")
        
        if st.button("Realizar predicción", key="predict_button"):
            model, predictions, mse = predict_data(df, feature_column, target_column, model_type)
            if model:
                st.write(f"Error cuadrático medio (MSE) del modelo {model_type}: {mse}")
                st.write("Resultados de las predicciones:")
                st.dataframe(predictions)
                
                fig, ax = plt.subplots()
                predictions.plot(kind='line', ax=ax)
                ax.set_title(f"Predicciones usando {model_type}")
                ax.set_xlabel("Índice")
                ax.set_ylabel(target_column)
                st.pyplot(fig)

                buffer = save_graph(fig, format='png')
                if buffer:
                    filename = f"prediccion_{model_type}.png"
                    st.download_button(
                        label="Guardar gráfico de predicciones",
                        data=buffer,
                        file_name=filename,
                        mime="image/png"
                    )

if __name__ == "__main__":
    main()
