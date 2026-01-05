class AppState {
    constructor({
                    modelName,
                    availableEpochs,
                    imageSize,
                    latentSpaceSize,
                    maxValueVisualizationInput,
                }) {
        this.modelName = modelName;
        this.availableEpochs = availableEpochs;
        this.imageSize = imageSize;
        this.latentSpaceSize = latentSpaceSize;
        this.latentSpaceSizeSqrt = latentSpaceSize ** 0.5;
        this.maxValueVisualizationInput = maxValueVisualizationInput;
        this.latentVector = new Array(latentSpaceSize).fill(0);
        this.inputValuesGenerator = null;
        this.inputValuesDiscriminator = null;
    }
}

export default AppState;