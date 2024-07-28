import streamlit as st
import pandas as pd
import io

def initialize_session_state():
    if 'combined_df' not in st.session_state:
        st.session_state.combined_df = None
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    if 'invalid_df' not in st.session_state:
        st.session_state.invalid_df = None

def main():
    initialize_session_state()
    
    st.title("Extractor y validador de datos Excel")
    
    uploaded_files = st.file_uploader("Selecciona archivos Excel", type="xlsx", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Procesar archivos"):
            process_files(uploaded_files)
    
    if st.session_state.combined_df is not None:
        st.dataframe(st.session_state.combined_df)
        if st.button("Validar datos"):
            validate_data()
    
    if st.session_state.filtered_df is not None:
        st.dataframe(st.session_state.filtered_df)
        if st.button("Guardar archivo procesado"):
            save_file()

def process_files(files):
    num_files = len(files)
    if num_files == 0:
        st.info("No se encontraron archivos Excel seleccionados.")
        return
    
    progress_bar = st.progress(0)
    data_frames = []
    
    for idx, file in enumerate(files):
        df = process_file(file, idx + 1, num_files, progress_bar)
        if df is not None:
            data_frames.append(df)
    
    if data_frames:
        st.session_state.combined_df = pd.concat(data_frames, ignore_index=True)
        st.success("Proceso completado.")
    else:
        st.error("No se pudo combinar ningún archivo.")

def process_file(file, file_index, num_files, progress_bar):
    try:
        df = pd.read_excel(file, engine='openpyxl', header=None)
        header_search = ['OFICINA', 'CÓDIGO', 'NOMBRE', 'LINEA', 'GRUPO', 'PNG', 'U', 'VALOR', 'U', 'VALOR', '% LV', 'VALOR (Más Cheque)', '%LV', 'U', 'VALOR', '% LC', 'VALOR (Más Cheque)', '% LC']

        header_row_index = df.apply(lambda row: all(keyword in row.values for keyword in header_search), axis=1)
        if not header_row_index.any():
            st.info(f"Encabezado no encontrado en el archivo {file.name}.")
            return None

        header_row_index = header_row_index.idxmax()
        header = df.iloc[header_row_index].tolist()
        data_df = df.iloc[header_row_index + 1:]
        data_df.columns = handle_duplicate_columns(header)

        progress_bar.progress(file_index / num_files)

        return data_df
    except Exception as e:
        st.error(f"Se produjo un error al procesar el archivo {file.name}: {e}")
        return None

def handle_duplicate_columns(columns):
    from collections import Counter

    col_counts = Counter(columns)
    for col, count in col_counts.items():
        if count > 1:
            indices = [i for i, x in enumerate(columns) if x == col]
            for i in range(1, count):
                columns[indices[i]] = f"{col}_{i}"
    return columns

def validate_data():
    try:
        valid_columns = ['OFICINA', 'CÓDIGO', 'NOMBRE', 'LINEA', 'GRUPO', 'PNG', 'U', 'VALOR', 'U_1', 'VALOR_1', '% LV', 'VALOR (Más Cheque)', '%LV', 'U_2', 'VALOR_2', '% LC', 'VALOR (Más Cheque)_1', '% LC_1']
        valid_rows = []
        invalid_rows = []

        for index, row in st.session_state.combined_df.iterrows():
            if row.notnull().all() and row[valid_columns].astype(str).apply(lambda x: x.strip() != "").all():
                valid_rows.append(row)
            else:
                invalid_rows.append(row)

        st.session_state.filtered_df = pd.DataFrame(valid_rows)
        st.session_state.invalid_df = pd.DataFrame(invalid_rows)
        st.success("Validación completada.")
    except Exception as e:
        st.error(f"Se produjo un error durante la validación: {e}")

def save_file():
    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            st.session_state.filtered_df.to_excel(writer, sheet_name='Sheet1', index=False)
            st.session_state.invalid_df.to_excel(writer, sheet_name='Sheet2', index=False)
        buffer.seek(0)

        st.download_button(
            label="Descargar archivo procesado",
            data=buffer,
            file_name="archivo_procesado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Se produjo un error al guardar el archivo: {e}")

if __name__ == "__main__":
    main()
