class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async synchronizeServer(modelName, latentSpaceSize) {
        console.log('==> Sync with server')
        try {
            const response = await fetch(`${this.baseUrl}/sync-server`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    model_size: modelName,
                    latent_space_size: latentSpaceSize,
                }),
            });

            return await response.json();
        } catch (error) {
            console.error("Error:", error);
        }
        return null;
    }

    async getModelPrediction(input_data, which_model, layer_name) {
        console.log('==> Get Model ',which_model,layer_name,'Prediction')
        try {
            const response = await fetch(`${this.baseUrl}/get-model-prediction`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    input_data: input_data,
                    which_model: which_model,
                    layer_name: layer_name}),
            });

            const response_content = await response.json();
            return response_content.output_values

        } catch (error) {
            console.error("Error:", error);
        }
        return null;
    }

    async changeEpoch(modelType, newEpoch) {
        console.log('==> Change Epoch',modelType,newEpoch);
        try {
            const response = await fetch(`${this.baseUrl}/change-epoch`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    new_epoch: newEpoch,
                    which_model: modelType

                }),
            });

            const response_content = await response.json();
            // todo add error if not changed
            return response_content.new_epoch_found;
        } catch (error) {
            console.error("Error:", error);
        }
        return null;
    }
}

export default ApiClient;