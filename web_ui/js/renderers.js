class ImageGridRenderer {
    initializeImage(elementId, sizeX, sizeY) {
        const divGrid = document.getElementById(elementId);
        console.log("init ",elementId)
        divGrid.classList.add("image-grid");
        divGrid.style.display = "grid";
        divGrid.style.gridTemplateColumns = `repeat(${sizeY}, 1fr)`;
        divGrid.style.gridTemplateRows = `repeat(${sizeX}, 1fr)`;

        const locationImage = Array.from({length: sizeX}, () => Array(sizeY).fill(null));

        for (let i = 0; i < sizeX; i++) {
            for (let j = 0; j < sizeY; j++) {
                const newElement = document.createElement("div");
                newElement.classList.add("slider_input_representation");
                divGrid.appendChild(newElement);
                newElement.style.width = "100%";
                newElement.style.height = "100%";

                locationImage[i][j] = newElement;
            }
        }

        return locationImage;
    }

    changeImage(newData, location) { //todo use this function to color grey input in generator
        //todo either this function should use data in the renderer class or moved out the class
        for (let i = 0; i < newData.length; i++) {
            for (let j = 0; j < newData[0].length; j++) {
                if (newData[i][j]) {
                    const [r, g, b] = newData[i][j];
                    location[i][j].style.backgroundColor = `rgb(${r}, ${g}, ${b})`;

                }
            }
        }
    }
}

class SliderGridRenderer {
    initializeGeneratorSliders(elementId, size, onInput) {
        const divSliders = document.getElementById(elementId);

        divSliders.style.gridTemplateColumns = `repeat(${size}, minmax(0, 1fr))`;
        divSliders.style.gridTemplateRows = `repeat(${size}, auto)`;

        const slidersGrid = Array.from({length: size}, () => Array(size).fill(null));

        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                const newElement = document.createElement("input");

                newElement.type = "range";
                newElement.min = -5;
                newElement.max = 5;
                newElement.step = 0.01;
                newElement.value = 0;
                newElement.classList.add("slider");

                newElement.oninput = function () {
                    onInput(i, j, newElement.value);
                };

                divSliders.appendChild(newElement);
                slidersGrid[i][j] = newElement;
            }
        }

        return slidersGrid;
    }

    refreshSliders(slidersGrid, latentVector, size) {
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                slidersGrid[i][j].value = latentVector[i * size + j];
            }
        }
    }
}

export {ImageGridRenderer, SliderGridRenderer};