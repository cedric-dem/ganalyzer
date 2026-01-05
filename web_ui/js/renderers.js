
class ImageGridRenderer {
    initializeImage(elementId, sizeX, sizeY) {
        const divGrid = document.getElementById(elementId);

        const availableWidth = divGrid.clientWidth;
        const availableHeight = divGrid.clientHeight;

        const pixelSize = Math.floor(Math.min(availableWidth / sizeY, availableHeight / sizeX));

        divGrid.style.display = "grid";
        divGrid.style.gridTemplateColumns = `repeat(${sizeY}, ${pixelSize}px)`;
        divGrid.style.gridTemplateRows = `repeat(${sizeX}, ${pixelSize}px)`;

        const locationImage = Array.from({length: sizeX}, () => Array(sizeY).fill(null));

        for (let i = 0; i < sizeX; i++) {
            for (let j = 0; j < sizeY; j++) {
                const newElement = document.createElement("div");
                newElement.classList.add("slider_input_representation");
                divGrid.appendChild(newElement);
                newElement.style.width = `${pixelSize}px`;
                newElement.style.height = `${pixelSize}px`;

                locationImage[i][j] = newElement;
            }
        }

        return locationImage;
    }

    changeImage(newData, location) {
        for (let i = 0; i < newData.length; i++) {
            for (let j = 0; j < newData[0].length; j++) {
                const [r, g, b] = newData[i][j];
                location[i][j].style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
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