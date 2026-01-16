<template>
  <div class="box">
    <h1 class="top_bar">Generator Visualization</h1>
    <div class="panel-row">
      <div class="small-panel">
        <div class="small_inside_panel_quarter_title center">
          <p>Epoch Slider</p>
        </div>
        <div class="small_inside_panel_quarter">
          <div class="card-surface center">
            <p id="labelGeneratorEpochValue"> Current Epoch : 0 / 100 </p>
          </div>
          <div class="card-surface center">
            <input
                id="sliderGeneratorEpochValue"
                type="range"
                min="0"
                max="200"
                value="95"
                step="1"
                @input="handleSliderGeneratorEpochValue($event.target.value)"
            />
          </div>
        </div>
      </div>

      <div class="small-panel">
        <div class="small_inside_panel_quarter_title center">
          <p>Input Generator</p>
        </div>
        <div class="small_inside_panel_quarter center">
          <div id="div_visualization_input_generator"></div>
        </div>
      </div>

      <div class="small-panel">
        <div class="small_inside_panel_quarter_title center">
          <p>Inside Generator</p>
        </div>

        <div class="small_inside_panel_quarter inside-visualization">
          <div class="card-surface label-select-row">
            <label for="choice_layer_generator">Choose a layer to visualize:</label>
            <select id="choice_layer_generator"></select>
          </div>

          <div class="card-surface center">
            <div id="div_visualization_inside_generator"></div>
          </div>
        </div>
      </div>

      <div class="small-panel">
        <div class="small_inside_panel_quarter_title center">
          <p>Output Generator</p>
        </div>
        <div class="small_inside_panel_quarter center">
          <div id="div_visualization_output_generator"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  handleSliderGeneratorEpochValue: (value: string) => void;
}>();
</script>