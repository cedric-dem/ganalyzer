//todo : why a generate calls generate twice

import CONFIG from "./js/config.js";
import WebUI from "./js/webUI.js";

const webUI = new WebUI(CONFIG);

webUI.initialize();