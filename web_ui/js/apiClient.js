class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async synchronizeServer(modelName, latentSpaceSize) {
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

    async getResultGenerator(vector) {
        try {
            const response = await fetch(`${this.baseUrl}/get-result-generator`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({vector: vector}),
            });

            return await response.json();
        } catch (error) {
            console.error("Error:", error);
        }
        return null;
    }

    async getInsideValues(vector, which_model, layer_name) {
        try {
            const response = await fetch(`${this.baseUrl}/get-inside-values`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({vector: vector, which_model: which_model,  layer_name: layer_name}),
            });

            const result =  await response.json();
            return result.inside_values

        } catch (error) {
            console.error("Error:", error);
        }
        return null;
    }

    async changeEpoch(modelType, newEpoch) {
        try {
            const response = await fetch(`${this.baseUrl}/change-epoch-${modelType}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({new_epoch: newEpoch}),
            });

            const data = await response.json();
            // todo add error if not changed
            return data.new_epoch_found;
        } catch (error) {
            console.error("Error:", error);
        }
        return null;
    }
}

export default ApiClient;