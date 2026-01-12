function addChoices(controller, isGenerator, location, layersList) { //todo coulr remove boolean isgenerator
    const select = document.getElementById(location);

    layersList.forEach((layer) => {
        const option = new Option(layer);
        select.add(option);
    });

    select.addEventListener("change", () => {
        const value = select.value;

        if (isGenerator) { //todo refactor code, both controller should inherit from controler abstract class so no if needed here
            controller.refreshInsideGeneratorNew(value)
        } else {
            controller.refreshInsideDiscriminatorNew(value)
        }
    });
}

async function updateGeneratorEpoch(newEpoch, state, apiClient, generatorController, shouldRefresh = true) {
    //send message to python api
    const foundEpoch = await apiClient.changeEpoch("generator", newEpoch);

    //change text
    document.getElementById("labelGeneratorEpochValue").textContent =
        "Epoch : " + newEpoch + "(" + foundEpoch + ")" + "/" + state.availableEpochs;
    if (shouldRefresh) {
        generatorController.refreshGeneratorAndDiscriminator();
    }
}

async function updateDiscriminatorEpoch(discriminatorController, newEpoch, state, apiClient, shouldRefresh = true) {
    //send message to python api
    const foundEpoch = await apiClient.changeEpoch("discriminator", newEpoch);

    //change text
    document.getElementById("labelDiscriminatorEpochValue").textContent =
        "Epoch : " + newEpoch + "(" + foundEpoch + ")" + "/" + state.availableEpochs;
    //get_result_discriminator()

    if (shouldRefresh) {
        //discriminatorController.refreshDiscriminator(//TODO);
    }
}

export {addChoices, updateDiscriminatorEpoch, updateGeneratorEpoch};