import CONFIG from "./src/js/config.js";
import WebUI from "./src/js/webUI.js";

const webUI = new WebUI(CONFIG);

webUI.initialize();