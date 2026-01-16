import { get1DNullArray, getRandomNormalFloat } from "../misc";
import { SliderRenderer } from "../renderer/sliderRenderer";
import WebUI from "../webUI";


export default class InputDataController {
    private currentLatentVector: number[];
    private callingWebUI: WebUI;
    private latentSpaceSize: number;
    private latentSpaceSizeSqrt: number;
    private sliderMuTextValue: HTMLElement;
    private sliderSigmaTextValue: HTMLElement;
    private sliderConstantTextValue: HTMLElement;
    private sliderMuValue: HTMLInputElement;
    private sliderSigmaValue: HTMLInputElement;
    private sliderConstantValue: HTMLInputElement;
    private sliderGridRenderer: SliderRenderer;

    constructor(callingWebUI: WebUI, latentSpaceSize: number, latentSpaceSizeSqrt: number) {
        this.currentLatentVector = get1DNullArray(latentSpaceSize);

        this.callingWebUI = callingWebUI;

        this.latentSpaceSize = latentSpaceSize;
        this.latentSpaceSizeSqrt = latentSpaceSizeSqrt;

        this.sliderMuTextValue = document.getElementById("sliderMuValueLabel") as HTMLElement;
        this.sliderSigmaTextValue = document.getElementById("sliderSigmaValueLabel") as HTMLElement;
        this.sliderConstantTextValue = document.getElementById("sliderConstantValueLabel") as HTMLElement;

        this.sliderMuValue = document.getElementById("sliderMuValue") as HTMLInputElement;
        this.sliderSigmaValue = document.getElementById("sliderSigmaValue") as HTMLInputElement;
        this.sliderConstantValue = document.getElementById("sliderConstantValue") as HTMLInputElement;

        this.sliderGridRenderer = new SliderRenderer("sliders_grid");
        this.sliderGridRenderer.initializeGeneratorSliders(
            latentSpaceSizeSqrt,
            (i, j, newValue) => this.handleSliderValueChange(i, j, newValue),
        );
    }

    getLatentVector(): number[] {
        return this.currentLatentVector;
    }

    handleSliderMuValue(value: string | number): void {
        this.sliderMuTextValue.textContent = String(value);
        this.randomizeInput();
    }

    handleSliderSigmaValue(value: string | number): void {
        this.sliderSigmaTextValue.textContent = String(value);
        this.randomizeInput();
    }

    handleSliderConstantValue(value: string | number): void {
        this.sliderConstantTextValue.textContent = String(value);
        this.setConstantInput();
    }

    getMuValue(): number {
        return parseFloat(this.sliderMuValue.value);
    }

    getSigmaValue(): number {
        return parseFloat(this.sliderSigmaValue.value);
    }

    getKValue(): number {
        return parseFloat(this.sliderConstantValue.value);
    }

    randomizeInput(): void {
        const mu = this.getMuValue();
        const sigma = this.getSigmaValue();

        for (let i = 0; i < this.latentSpaceSize; i++) {
            this.currentLatentVector[i] = getRandomNormalFloat(mu, sigma);
        }
        this.refreshSliders();
        this.refreshGenerator();
    }

        setConstantInput(): void {
            const k = this.getKValue();
            for (let i = 0; i < this.currentLatentVector.length; i++) {
            this.currentLatentVector[i] = k;
        }
        this.refreshSliders();
        this.refreshGenerator();
    }

        refreshSliders(): void {
            this.sliderGridRenderer.refreshSliders(this.currentLatentVector, this.latentSpaceSizeSqrt);
        }

        handleSliderValueChange(i: number, j: number, newValue: number | string): void {
            const parsedValue = typeof newValue === "number" ? newValue : parseFloat(newValue);
            this.currentLatentVector[i * this.latentSpaceSizeSqrt + j] = parsedValue;
            this.refreshGenerator();
        }

        refreshGenerator(): void {
            this.callingWebUI.refreshGenerator();
        }
    }