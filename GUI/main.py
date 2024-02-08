import gradio as gr
import numpy as np
import joblib

model = joblib.load('CatBoost.joblib')

sex_choice = ['Male', 'Female']
marital_choice = ['Single', 'Married', 'Widowed', 'Divorced', 'NaN', 'Separated']
race_choice = ['White', 'Asian', 'Black', 'MexAmerican', 'Hispanic', 'Other']
def process_data(seqn, Age, Sex, Marital, Income, Race, WaistCirc, BMI, Albuminuria,
                 UrAlbCr, UricAcid, BloodGlucose, HDL, Triglycerides):
    try:

        input_data_encoded = [[seqn, Age, Sex, Marital, Income,  Race, WaistCirc, BMI, Albuminuria,
                 UrAlbCr, UricAcid, BloodGlucose, HDL, Triglycerides]]

        y_pred_prob = model.predict_proba(input_data_encoded)
        y_pred = np.where(y_pred_prob[:, 1] > 0.53, 1, 0)
        result_text = "Вам стоит обратиться к врачу, возможно у вас метаболический синдром!" if y_pred == 1 else "С вами всё хорошо, кушайте пончики и дальше!"

        return result_text


    except ValueError as e:
        print(f"Exception: {e}")
        result_text = "Некорректный формат данных."
        return result_text

iface = gr.Interface(
    fn=process_data,
    inputs=[
        # Отсутствие синдрома
        gr.Number(label="Введите seqn", value=66040),
        gr.Number(label="Введите Age", value=24),
        gr.Dropdown(label="Выберите Sex", choices=sex_choice, value='Female'),
        gr.Dropdown(label="Выберите Marital", choices=marital_choice, value='NaN'),
        gr.Number(label="Введите Income", value=2500),
        gr.Dropdown(label="Выберите Race", choices=race_choice, value='Hispanic'),
        gr.Number(label="Введите WaistCirc", value=77.8),
        gr.Number(label="Введите BMI", value=26.2),
        gr.Number(label="Введите Albuminuria", value=0),
        gr.Number(label="Введите UrAlbCr", value=23.67),
        gr.Number(label="Введите UricAcid", value=3.7),
        gr.Number(label="Введите BloodGlucose", value=82),
        gr.Number(label="Введите HDL", value=55),
        gr.Number(label="Введите Triglycerides", value=119),

        # Наличие синдрома
        # gr.Number(label="Введите seqn", value=67598),
        # gr.Number(label="Введите Age", value=80),
        # #gr.Textbox(label="Выберите Sex"),
        # #gr.Textbox(label="Выберите Sex"),
        # gr.Dropdown(label="Выберите Sex", choices=sex_choice, value='Female'),
        # gr.Dropdown(label="Выберите Marital", choices=marital_choice, value='Divorced'),
        # gr.Number(label="Введите Income", value=6200),
        # #gr.Textbox(label="Выберите Sex"),
        # gr.Dropdown(label="Выберите Race", choices=race_choice, value='Hispanic'),
        # gr.Number(label="Введите WaistCirc", value=99.5),
        # gr.Number(label="Введите BMI", value=28.5),
        # gr.Number(label="Введите Albuminuria", value=0),
        # gr.Number(label="Введите UrAlbCr", value=20.52),
        # gr.Number(label="Введите UricAcid", value=8.4),
        # gr.Number(label="Введите BloodGlucose", value=107),
        # gr.Number(label="Введите HDL", value=42),
        # gr.Number(label="Введите Triglycerides", value=193),
    ],
    outputs=gr.Textbox(),
)

if __name__ == "__main__":
    iface.launch()
