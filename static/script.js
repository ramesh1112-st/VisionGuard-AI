// static/script.js

const videoInput = document.getElementById('videoInput');
const videoPreview = document.getElementById('videoPreview');
const previewBox = document.getElementById('previewBox');

videoInput.addEventListener('change', function(){

    const file = this.files[0];

    if(file){

        const videoURL = URL.createObjectURL(file);

        videoPreview.src = videoURL;

        previewBox.style.display = "block";
    }

});