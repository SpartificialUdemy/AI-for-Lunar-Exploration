import streamlit as st
from utils import predict, lunar_model, load_image, colorful_mask, load_model

def main():
    """
    Main function for the Streamlit app.
    """
    # Title of the app
    st.markdown(
        "<h1 style='color: cyan;'>Lunar Surface Segmentation App</h1>",
        unsafe_allow_html=True
    )

    # Upload image section
    uploaded_file = st.file_uploader("Upload an image of the lunar surface to segment it...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Displaying image loading message
        st.write('Loading the Image...')
        
        # Preprocess the uploaded image
        img_out = load_image(uploaded_file)
        
        # Load the pre-trained Lunar Model
        loaded_model = load_model('model/LunarModel.h5')

        # Display uploaded image
        st.markdown(
            "<h2 style='color: yellow;'>Input Image - </h2>",
            unsafe_allow_html=True
        )
        st.image(img_out, use_column_width=True)
        
        if st.button("Segment the Lunar Image"):
            # Segmenting image
            st.write("Segmentation in progress...")
            pred_mask = predict(img_out, loaded_model)
            pred_mask_colored = colorful_mask(pred_mask)

            # Display segmented image
            st.markdown(
                "<h2 style='color: magenta;'>Predicted Mask - </h2>",
                unsafe_allow_html=True
            )
            st.image(pred_mask_colored, use_column_width=True)
            st.write("Segmentation complete.")

            # Displaying thank you message
            st.markdown(
                "<h1 style='color: orange;'>Thanks for using our App!</h1>",
                unsafe_allow_html=True
            )
            
            st.write("Add more images to browse section to segment more Lunar images! ^_^")

# Run the app
if __name__ == "__main__":
    main()
