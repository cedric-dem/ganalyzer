<template>
  <div class="box">
    <h1 class="top_bar">Generator Input</h1>

    <div class="panel-row">
      <div class="control-card">
        <div class="control-card__header">Set Random Values</div>
        <div class="control-card__content">
          <div class="slider-stack">
            <label for="sliderMuValue">Mean (μ)</label>
            <input
                id="sliderMuValue"
                type="range"
                min="-10"
                max="10"
                value="0"
                step="0.01"
                @change="handleSliderMuValue($event.target.value)"
            />
            <p class="value-label">mu = <span id="sliderMuValueLabel">0</span></p>
          </div>
          <div class="slider-stack">
            <label for="sliderSigmaValue">Std Dev (σ)</label>
            <input
                id="sliderSigmaValue"
                type="range"
                min="0"
                max="5"
                value="1"
                step="0.01"
                @change="handleSliderSigmaValue($event.target.value)"
            />
            <p class="value-label">sigma = <span id="sliderSigmaValueLabel">1</span></p>
          </div>
          <button class="primary-btn" type="button" @click="reRandomize">Regenerate</button>
        </div>
      </div>

      <div class="control-card">
        <div class="control-card__header">Set All Values Constant</div>
        <div class="control-card__content">
          <div class="slider-stack">
            <label for="sliderConstantValue">Constant value</label>
            <input
                id="sliderConstantValue"
                type="range"
                min="-5"
                max="5"
                value="0"
                step="0.01"
                @change="handleSliderConstantValue($event.target.value)"
            />
            <p class="value-label">k = <span id="sliderConstantValueLabel">0</span></p>
          </div>
          <button class="primary-btn" type="button" @click="setConstantInput">Set</button>
        </div>
      </div>
    </div>

    <div class="panel-block">
      <div class="panel-block__inner">
        <div id="sliders_grid" class="card-surface"></div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  handleSliderMuValue: (value: string) => void;
  handleSliderSigmaValue: (value: string) => void;
  handleSliderConstantValue: (value: string) => void;
  reRandomize: () => void;
  setConstantInput: () => void;
}>();
</script>