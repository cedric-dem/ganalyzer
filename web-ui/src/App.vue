<template>
  <div class="container">
    <GeneratorInputPanel
        :handle-slider-mu-value="handleSliderMuValue"
        :handle-slider-sigma-value="handleSliderSigmaValue"
        :handle-slider-constant-value="handleSliderConstantValue"
        :re-randomize="reRandomize"
        :set-constant-input="setConstantInput"
    />

    <GeneratorVisualizationPanel
        :handle-slider-generator-epoch-value="handleSliderGeneratorEpochValue"
    />

    <DiscriminatorVisualizationPanel
        :handle-slider-discriminator-epoch-value="handleSliderDiscriminatorEpochValue"
    />
  </div>
</template>

<script setup>
import { onMounted } from "vue";
import GeneratorInputPanel from "./components/GeneratorInputPanel.vue";
import GeneratorVisualizationPanel from "./components/GeneratorVisualizationPanel.vue";
import DiscriminatorVisualizationPanel from "./components/DiscriminatorVisualizationPanel.vue";
import CONFIG from "./js/config.js";
import WebUI from "./js/webUI.js";

let webUI = null;

const handleSliderMuValue = (value) => {
  if (webUI) {
    webUI.inputDataController.handleSliderMuValue(value);
  }
};

const handleSliderSigmaValue = (value) => {
  if (webUI) {
    webUI.inputDataController.handleSliderSigmaValue(value);
  }
};

const handleSliderConstantValue = (value) => {
  if (webUI) {
    webUI.inputDataController.handleSliderConstantValue(value);
  }
};

const reRandomize = () => {
  if (webUI) {
    webUI.inputDataController.randomizeInput();
  }
};

const setConstantInput = () => {
  if (webUI) {
    webUI.inputDataController.setConstantInput();
  }
};

const handleSliderGeneratorEpochValue = (value) => {
  if (webUI) {
    webUI.generatorController.updateEpoch(value);
  }
};

const handleSliderDiscriminatorEpochValue = (value) => {
  if (webUI) {
    webUI.discriminatorController.updateEpoch(value);
  }
};

onMounted(() => {
  webUI = new WebUI(CONFIG);
  webUI.initialize();
});
</script>
