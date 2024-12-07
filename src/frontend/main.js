// main.js
document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const carPhotoInput = document.getElementById('carPhoto');
    const predictionResult = document.getElementById('predictionResult');
    const MAX_SIZE = 1.5 * 1024 * 1024; // 1.5MB in bytes

    // Add a hint message beneath the file input
    const hint = document.createElement('p');
    hint.style.color = 'gray';
    hint.textContent = 'Hint: Please upload an image smaller than 1.5MB.';
    carPhotoInput.parentNode.insertBefore(hint, carPhotoInput.nextSibling);

    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        console.log('Form submitted');

        // Check if a file is selected
        if (carPhotoInput.files.length === 0) {
            alert('Please select a photo to upload.');
            return;
        }

        // Check the file size before uploading
        const file = carPhotoInput.files[0];
        if (file.size > MAX_SIZE) {
            alert('The selected image is larger than 1.5MB. Please choose a smaller file.');
            return;
        }

        // Proceed with the upload if size is valid
        const formData = new FormData();
        formData.append('image', file);
        console.log('File appended to FormData');

        try {
            // const response = await fetch('http://localhost:9000/predict', {
            const response = await fetch('/api/predict', {
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
