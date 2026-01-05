function addChoices(discriminatorController, isGenerator, location, layersList) {
    const select = document.getElementById(location);

    layersList.forEach((layer) => {
        const option = new Option(layer);
        select.add(option);
    });

    select.addEventListener("change", () => {
        const value = select.value;
        discriminatorController.changeInsideVisualization(isGenerator, value);
    });
}

async function updateGeneratorEpoch(newEpoch, state, apiClient, generatorController) {
    //send message to python api
    const foundEpoch = await apiClient.changeEpoch("generator", newEpoch);

    //change text
    document.getElementById("labelGeneratorEpochValue").textContent =
        "Epoch : " + newEpoch + "(" + foundEpoch + ")" + "/" + state.availableEpochs;
    generatorController.submitToApi();
}

async function updateDiscriminatorEpoch(newEpoch, state, apiClient) {
    //send message to python api
    const foundEpoch = await apiClient.changeEpoch("discriminator", newEpoch);

    //change text
    document.getElementById("labelDiscriminatorEpochValue").textContent =
        "Epoch : " + newEpoch + "(" + foundEpoch + ")" + "/" + state.availableEpochs;
    //get_result_discriminator()
}

export {addChoices, updateDiscriminatorEpoch, updateGeneratorEpoch};