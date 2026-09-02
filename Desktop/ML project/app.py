import streamlit as st
import pickle
import re
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# Find Model File
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


model = load_model()


# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🤖 Customer Review Sentiment Analyzer")

st.write(
    "Enter a customer review and the machine learning "
    "model will predict whether it is positive or negative."
)

st.divider()


# --------------------------------------------------
# Check Model
# --------------------------------------------------

if model is None:

    st.error(
        "❌ Machine learning model not found."
    )

    st.info(
        "Please run 'python train_model.py' first."
    )

    st.code(
        "python train_model.py\n"
        "streamlit run app.py"
    )

    st.stop()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("About")

st.sidebar.write(
    """
    ### Machine Learning

    This application uses:

    - Python
    - Streamlit
    - TF-IDF
    - Logistic Regression
    - Natural Language Processing

    The model classifies customer reviews
    as Positive or Negative.
    """
)


# --------------------------------------------------
# User Input
# --------------------------------------------------

st.subheader("📝 Enter Customer Review")

review = st.text_area(
    "Customer Review",
    placeholder=(
        "Example: The product is excellent "
        "and I really enjoyed using it!"
    ),
    height=150
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "🔍 Analyze Review",
    type="primary"
):

    if not review.strip():

        st.warning(
            "⚠️ Please enter a review."
        )

    else:

        cleaned_review = clean_text(review)

        prediction = model.predict(
            [cleaned_review]
        )[0]

        probabilities = model.predict_proba(
            [cleaned_review]
        )[0]

        # Get class names from model
        classes = model.classes_

        probability_dict = {
            classes[i]: probabilities[i] * 100
            for i in range(len(classes))
        }

        confidence = max(probabilities) * 100


        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        st.divider()

        st.subheader("🎯 Prediction")

        if prediction == "Positive":

            st.success(
                "😊 Positive Review"
            )

        else:

            st.error(
                "😞 Negative Review"
            )


        # --------------------------------------------------
        # Confidence
        # --------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Prediction",
                prediction
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


        # --------------------------------------------------
        # Original Review
        # --------------------------------------------------

        st.subheader("📄 Your Review")

        st.info(review)


        # --------------------------------------------------
        # Probability
        # --------------------------------------------------

        st.subheader(
            "📊 Prediction Probability"
        )

        st.bar_chart(
            probability_dict
        )


# --------------------------------------------------
# Example Reviews
# --------------------------------------------------

st.divider()

st.subheader("💡 Try These Examples")

examples = [
    "The product is amazing and the quality is excellent.",
    "I am very disappointed with this product.",
    "The service was fast and helpful.",
    "This product is terrible and a waste of money.",
    "I am very happy with my purchase."
]

for example in examples:

    st.write(
        "•",
        example
    )
