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


export {addChoices};