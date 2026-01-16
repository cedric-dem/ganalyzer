<template>
  <div class="box">
    <h1 class="top_bar">Discriminator Visualization</h1>

    <div class="panel-row">
      <div class="small-panel">
        <div class="small_inside_panel_quarter_title center">
          <p>Epoch Slider</p>
        </div>
        <div class="small_inside_panel_quarter">
          <div class="card-surface center">
            <p id="labelDiscriminatorEpochValue"> Current Epoch : 0 / 100 </p>
          </div>
          <div class="card-surface center">
            <input
                id="sliderDiscriminatorEpochValue"
                type="range"
                min="0"
                max="200"
                value="95"
                step="1"
                @input="handleSliderDiscriminatorEpochValue($event.target.value)"
            />
          </div>
        </div>
      </div>

      <div class="small-panel">
        <div class="small_inside_panel_quarter_title center">
          <p>Input Discriminator</p>
        </div>
        <div class="small_inside_panel_quarter center">
          <div id="div_visualization_input_discriminator"></div>
        </div>
      </div>

      <div class="small-panel">
        <div class="small_inside_panel_quarter_title center">
          <p>Inside Discriminator</p>
        </div>

        <div class="small_inside_panel_quarter inside-visualization">
          <div class="card-surface label-select-row">
            <label for="choice_layer_discriminator">Choose a layer to visualize:</label>
            <select id="choice_layer_discriminator"></select>
          </div>

          <div class="card-surface center">
            <div id="div_visualization_inside_discriminator"></div>
          </div>
        </div>
      </div>

      <div class="small-panel">
        <div class="small_inside_panel_quarter_title center">
          <p>Output Discriminator</p>
        </div>
        <div class="small_inside_panel_quarter center">
          <p id="prediction_output_text">Prediction Incoming</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  handleSliderDiscriminatorEpochValue: (value: string) => void;
}>();
</script>