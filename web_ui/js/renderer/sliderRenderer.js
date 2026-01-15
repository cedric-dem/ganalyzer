import {get2DNullArray} from "../misc.js";

class SliderRenderer {
    constructor(elementId) {
        this.slidersGrid = null;
        this.divSliders = document.getElementById(elementId);
    }

    initializeGeneratorSliders(size, onInput) {

        //maybe not ?
        this.divSliders.innerHTML = "";

        this.divSliders.style.gridTemplateColumns = `repeat(${size}, minmax(0, 1fr))`;
        this.divSliders.style.gridTemplateRows = `repeat(${size}, auto)`;

        this.slidersGrid = get2DNullArray(size, size)

        let newElement = null;
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                newElement = document.createElement("input");

                newElement.type = "range";
                newElement.min = -5;
                newElement.max = 5;
                newElement.step = 0.01;
                newElement.classList.add("slider");

                newElement.oninput = function () {
                    onInput(i, j, newElement.value);
                };

                this.divSliders.appendChild(newElement);
                this.slidersGrid[i][j] = newElement;
            }
        }
    }

    refreshSliders(latentVector, size) {
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                this.slidersGrid[i][j].value = latentVector[i * size + j];
            }
        }
    }
}

export {SliderRenderer};