import {changeInsideRepresentation} from "./misc.js";

export class ModelController {
    constructor(callingWebUI, modelName, apiClient, locationInsideVisualization, locationEpochLabel) {

        this.callingWebUI = callingWebUI;
        this.modelName = modelName;
        this.apiClient = apiClient;

        this.inputData = null;

        this.locationInsideVisualization = locationInsideVisualization
        this.locationEpochLabel = document.getElementById(locationEpochLabel)

    }
    async refreshAll(){

    }

    async refreshInside(layerToVisualize) {

        //api call with the current layer and 'discriminator'
        const newInsideValues = await this.apiClient.getModelPrediction(this.inputData, this.modelName, layerToVisualize);

        //change image
        changeInsideRepresentation(newInsideValues, this.locationInsideVisualization)

    }

    initializeLastLayer(lastLayerName) {
        this.lastLayerName = lastLayerName
    }

    async updateEpoch(newEpoch, shouldRefresh = true) {

        //send message to python api
        const foundEpoch = await this.apiClient.changeEpoch(this.modelName, newEpoch);

        //change text
        this.locationEpochLabel.textContent = "Epoch : " + newEpoch + "(" + foundEpoch + ")" + "/" + this.callingWebUI.availableEpochs;

        if (shouldRefresh) {
            await this.refreshAll();
        }
    }
}