
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

MODEL_REPO = "navtri12/tourism-model"


@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="best_model.joblib",
        repo_type="model"
    )

    return joblib.load(model_path)


model = load_model()


st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Wellness Tourism Package Predictor")

st.write(
    "Enter customer details to predict whether the customer "
    "is likely to purchase the Wellness Tourism Package."
)


age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

typeof_contact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

occupation = st.selectbox(
    "Occupation",
    [
        "Salaried",
        "Free Lancer",
        "Small Business",
        "Large Business"
    ]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

num_persons = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=2
)

preferred_star = st.selectbox(
    "Preferred Property Star",
    [3.0, 4.0, 5.0]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

num_trips = st.number_input(
    "Number of Trips",
    min_value=0,
    max_value=20,
    value=2
)

passport = st.selectbox(
    "Passport",
    [0, 1]
)

own_car = st.selectbox(
    "Own Car",
    [0, 1]
)

num_children = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=5,
    value=0
)

designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=100000,
    value=20000
)

pitch_score = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

product_pitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Deluxe",
        "Standard",
        "Super Deluxe",
        "King"
    ]
)

num_followups = st.number_input(
    "Number of Followups",
    min_value=0,
    max_value=10,
    value=3
)

duration_of_pitch = st.number_input(
    "Duration of Pitch",
    min_value=1,
    max_value=60,
    value=10
)


if st.button("Predict Purchase"):

    input_data = pd.DataFrame([{

        "Age": age,
        "TypeofContact": typeof_contact,
        "CityTier": city_tier,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_persons,
        "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": passport,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": num_children,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
        "PitchSatisfactionScore": pitch_score,
        "ProductPitched": product_pitched,
        "NumberOfFollowups": num_followups,
        "DurationOfPitch": duration_of_pitch

    }])


    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]


    st.subheader("Prediction Result")


    if prediction == 1:

        st.success(
            "The customer is predicted to purchase "
            "the Wellness Tourism Package."
        )

    else:

        st.info(
            "The customer is predicted not to purchase "
            "the Wellness Tourism Package."
        )


    st.metric(
        "Purchase Probability",
        f"{probability:.2%}"
    )
