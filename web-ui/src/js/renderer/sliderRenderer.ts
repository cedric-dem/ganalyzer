import {getEmptyRGB2DImage} from "../misc";
import {numberVector} from "../types/types";

type SliderInputHandler = (row: number, column: number, value: string) => void;


export class SliderRenderer {
    private slidersGrid: any = [];
    private divSliders: HTMLElement;

    constructor(elementId: string) {
        const element = document.getElementById(elementId);
        if (!element) {
            throw new Error(`SliderRenderer: element with id "${elementId}" not found`);
        }
        this.divSliders = element;
    }

    initializeGeneratorSliders(size: number, onInput: SliderInputHandler): void {

        //maybe not ?
        this.divSliders.innerHTML = "";

        this.divSliders.style.gridTemplateColumns = `repeat(${size}, minmax(0, 1fr))`;
        this.divSliders.style.gridTemplateRows = `repeat(${size}, auto)`;

        this.slidersGrid = getEmptyRGB2DImage(size, size);

        let newElement: HTMLInputElement | null = null;
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                newElement = document.createElement("input");

                newElement.type = "range";
                newElement.min = "-5";
                newElement.max = "5";
                newElement.step = "0.01";
                newElement.classList.add("slider");

                newElement.oninput = function () {
                    onInput(i, j, newElement.value);
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