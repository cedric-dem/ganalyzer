import {getEmptyRGB2DImage} from "../misc";
import {numberVector} from "../types/types";

type SliderInputHandler = (row: number, column: number, value: number) => void;

export class SliderRenderer {
    private slidersGrid: any = [];
    private divSliders: HTMLElement;

    constructor(elementId: string) {
        this.divSliders = document.getElementById(elementId);
    }

    initializeGeneratorSliders(
        size: number,
        maxValueVisualizationInput: number,
        onInput: SliderInputHandler,
    ): void {

        //maybe not ?
        this.divSliders.innerHTML = "";

        this.divSliders.style.gridTemplateColumns = `repeat(${size}, minmax(0, 1fr))`;
        this.divSliders.style.gridTemplateRows = `repeat(${size}, auto)`;

        this.slidersGrid = getEmptyRGB2DImage(size, size);

        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                const newElement = document.createElement("input");

                newElement.type = "range";
                newElement.min = String(-maxValueVisualizationInput);
                newElement.max = String(maxValueVisualizationInput);
                newElement.step = "0.01";
                newElement.classList.add("slider");

                newElement.onchange = (event) => {
                    const target = event.currentTarget as HTMLInputElement;
                    onInput(i, j, parseFloat(target.value));
                };

                this.divSliders.appendChild(newElement);
                this.slidersGrid[i][j] = newElement;
            }
        }
    }


    refreshSliders(latentVector: numberVector, size: number): void {
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                const slider = this.slidersGrid[i][j];
                if (slider) {
                    slider.value = String(latentVector[i * size + j]);
                }
            }
        }
    }
}