class ImageRenderer {
    constructor(elementId) {
        this.divGrid = document.getElementById(elementId);
    }

    initializeImage(sizeX, sizeY) {
        //maybe not ?
        this.divGrid.innerHTML = "";

        this.divGrid.classList.add("image-grid");
        this.divGrid.style.display = "grid";
        this.divGrid.style.gridTemplateColumns = `repeat(${sizeY}, 1fr)`;
        this.divGrid.style.gridTemplateRows = `repeat(${sizeX}, 1fr)`;

        this.locationImage = Array.from({length: sizeX}, () => Array(sizeY).fill(null));

        let newElement = null;
        for (let i = 0; i < sizeX; i++) {
            for (let j = 0; j < sizeY; j++) {
                newElement = document.createElement("div");
                newElement.classList.add("slider_input_representation");
                newElement.style.width = "100%";
                newElement.style.height = "100%";

                this.divGrid.appendChild(newElement);
                this.locationImage[i][j] = newElement;
            }
        }
    }

    changeImage(newData) { //todo use this function to color grey input in generator
        //todo either this function should use data in the renderer class or moved out the class
        for (let i = 0; i < newData.length; i++) {
            for (let j = 0; j < newData[0].length; j++) {
                if (newData[i][j]) {
                    const [r, g, b] = newData[i][j];
                    this.locationImage[i][j].style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
                }
            }
        }
    }
}


export {ImageRenderer};