import {get1DNullArray, getRandomNormalFloat} from "../misc.js";
import {SliderRenderer} from "../renderer/sliderRenderer.js";

export default class InputDataController {
    constructor(callingWebUI, latentSpaceSize, latentSpaceSizeSqrt) {
        this.currentLatentVector = get1DNullArray(latentSpaceSize);

        this.callingWebUI = callingWebUI;

        this.latentSpaceSize = latentSpaceSize
        this.latentSpaceSizeSqrt = latentSpaceSizeSqrt

        this.sliderMuTextValue = document.getElementById("sliderMuValueLabel")
        this.sliderSigmaTextValue = document.getElementById("sliderSigmaValueLabel")
        this.sliderConstantTextValue = document.getElementById("sliderConstantValueLabel")

        this.sliderMuValue = document.getElementById("sliderMuValue")
        this.sliderSigmaValue = document.getElementById("sliderSigmaValue")
        this.sliderConstantValue = document.getElementById("sliderConstantValue")

        this.sliderGridRenderer = new SliderRenderer("sliders_grid");
        this.sliderGridRenderer.initializeGeneratorSliders(latentSpaceSizeSqrt, (i, j, newValue) => this.handleSliderValueChange(i, j, newValue));
    }

    getLatentVector() {
        return this.currentLatentVector;
    }

    handleSliderMuValue(value) {
        this.sliderMuTextValue.textContent = value;
        this.randomizeInput();
    }

    handleSliderSigmaValue(value) {
        this.sliderSigmaTextValue.textContent = value;
        this.randomizeInput();
    }

    handleSliderConstantValue(value) {
        this.sliderConstantTextValue.textContent = value;
        this.setConstantInput();
    }

    getMuValue() {
        return parseFloat(this.sliderMuValue.value)
    }

    getSigmaValue() {
        return parseFloat(this.sliderSigmaValue.value)
    }

    getKValue() {
        return parseFloat(this.sliderConstantValue.value);
    }

    randomizeInput() {
        const mu = this.getMuValue();
        const sigma = this.getSigmaValue();

        for (let i = 0; i < this.latentSpaceSize; i++) {
            this.currentLatentVector[i] = getRandomNormalFloat(mu, sigma);
        }
        this.refreshSliders();
        this.refreshGenerator();
    }

    setConstantInput() {
        const k = this.getKValue();
        for (let i = 0; i < this.currentLatentVector.length; i++) {
            this.currentLatentVector[i] = k;
        }
        this.refreshSliders();
        this.refreshGenerator();
    }

    refreshSliders() {
        this.sliderGridRenderer.refreshSliders(this.currentLatentVector, this.latentSpaceSizeSqrt);
    }

    handleSliderValueChange(i, j, newValue) {
        this.currentLatentVector[i * this.latentSpaceSizeSqrt + j] = newValue;
        this.refreshGenerator();
    }

    refreshGenerator(){
        this.callingWebUI.refreshGenerator();
    }
}