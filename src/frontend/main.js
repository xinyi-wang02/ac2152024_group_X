// main.js
document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const carPhotoInput = document.getElementById('carPhoto');
    const predictionResult = document.getElementById('predictionResult');

    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        console.log('Form submitted');

        if (carPhotoInput.files.length === 0) {
            alert('Please select a photo to upload.');
            return;
        }

        const formData = new FormData();
        formData.append('image', carPhotoInput.files[0]);
        console.log('File appended to FormData');

        try {
            const response = await fetch('http://localhost:9191/predict', {
                method: 'POST',
                body: formData,
            });
            console.log('Response received:', response);

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('Response data:', data);

            predictionResult.innerHTML = `<p>Predicted Car Model: ${data.predicted_car_types.join(', ')}</p>`;
        } catch (error) {
            console.error('Error:', error);
            predictionResult.innerHTML = `<p>Failed to get prediction. Please try again.</p>`;
        }
    });
});
