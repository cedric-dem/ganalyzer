import {RGB2DImage} from "../types/types";

export class ImageRenderer {
    private divGrid: HTMLDivElement;
    private locationImage: HTMLDivElement[][] = [];

    constructor(elementId: string) {
        const divGrid = document.getElementById(elementId);
        if (!divGrid) {
            throw new Error(`ImageRenderer element not found: ${elementId}`);
        }
        this.divGrid = divGrid as HTMLDivElement;
    }

    initializeImage(sizeX: number, sizeY: number): void {
        //maybe not ?
        this.divGrid.innerHTML = "";

        this.divGrid.classList.add("image-grid");
        this.divGrid.style.display = "grid";
        this.divGrid.style.gridTemplateColumns = `repeat(${sizeY}, 1fr)`;
        this.divGrid.style.gridTemplateRows = `repeat(${sizeX}, 1fr)`;

        this.locationImage = Array.from({length: sizeX}, () => Array(sizeY).fill(null)) as HTMLDivElement[][];

        let newElement: HTMLDivElement | null = null;
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

    changeImage(newData: RGB2DImage): void { //todo use this function to color grey input in generator
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