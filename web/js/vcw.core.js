import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function chainCallback(object, property, callback) {
  if (object == undefined) {
    //This should not happen.
    console.error("Tried to add callback to non-existant object");
    return;
  }
  if (property in object && object[property]) {
    const callback_orig = object[property];
    object[property] = function () {
      const r = callback_orig.apply(this, arguments);
      return callback.apply(this, arguments) ?? r;
    };
  } else {
    object[property] = callback;
  }
}

async function uploadFile(file) {
  try {
    // Wrap file in formdata so it includes filename
    const body = new FormData();
    const i = file.webkitRelativePath.lastIndexOf("/");
    const subfolder = file.webkitRelativePath.slice(0, i + 1);
    const new_file = new File([file], file.name, {
      type: file.type,
      lastModified: file.lastModified,
    });
    body.append("image", new_file);
    if (i > 0) {
      body.append("subfolder", subfolder);
    }
    const resp = await api.fetchApi("/upload/image", {
      method: "POST",
      body,
    });

    if (resp.status === 200) {
      return resp;
    } else {
      alert(resp.status + " - " + resp.statusText);
    }
  } catch (error) {
    alert(error);
  }
}

function addUploadWidget(nodeType, nodeData, widgetName, type = "video") {
  chainCallback(nodeType.prototype, "onNodeCreated", function () {
    const pathWidget = this.widgets.find((w) => w.name === widgetName);
    const fileInput = document.createElement("input");
    chainCallback(this, "onRemoved", () => {
      fileInput?.remove();
    });
    if (type == "video") {
      Object.assign(fileInput, {
        type: "file",
        accept: "video/webm, video/mp4, video/mov, video/mkv",
        style: "display: none",
        onchange: async () => {
          if (fileInput.files.length) {
            let resp = await uploadFile(fileInput.files[0]);
            if (resp.status != 200) {
              //upload failed and file can not be added to options
              return;
            }
            const filename = (await resp.json()).name;
            pathWidget.options.values.push(filename);
            pathWidget.value = filename;
            if (pathWidget.callback) {
              pathWidget.callback(filename);
            }
          }
        },
      });
    } else {
      throw "Unknown upload type";
    }
    document.body.append(fileInput);
    let uploadWidget = this.addWidget(
      "button",
      "choose " + type + " to upload",
      "image",
      () => {
        //clear the active click event
        app.canvas.node_widget = null;

        fileInput.click();
      }
    );
    uploadWidget.options.serialize = false;
  });
}

function fitHeight(node) {
  node.setSize([
    node.size[0],
    node.computeSize([node.size[0], node.size[1]])[1],
  ]);
  node?.graph?.setDirtyCanvas(true);
}

function startDraggingItems(node, pointer) {
  app.canvas.emitBeforeChange();
  app.canvas.graph?.beforeChange();
  // Ensure that dragging is properly cleaned up, on success or failure.
  pointer.finally = () => {
    app.canvas.isDragging = false;
    app.canvas.graph?.afterChange();
    app.canvas.emitAfterChange();
  };
  app.canvas.processSelect(node, pointer.eDown, true);
  app.canvas.isDragging = true;
}

function processDraggedItems(e) {
  if (e.shiftKey || LiteGraph.alwaysSnapToGrid)
    app.graph?.snapToGrid(app.canvas.selectedItems);
  app.canvas.dirty_canvas = true;
  app.canvas.dirty_bgcanvas = true;
  app.canvas.onNodeMoved?.(findFirstNode(app.canvas.selectedItems));
}

function allowDragFromWidget(widget) {
  widget.onPointerDown = function (pointer, node) {
    pointer.onDragStart = () => startDraggingItems(node, pointer);
    pointer.onDragEnd = processDraggedItems;
    app.canvas.dirty_canvas = true;
    return true;
  };
}

function addVideoPreview(nodeType, isInput = true) {
  chainCallback(nodeType.prototype, "onNodeCreated", function () {
    var element = document.createElement("div");
    const previewNode = this;
    var previewWidget = this.addDOMWidget("videopreview", "preview", element, {
      serialize: false,
      hideOnZoom: false,
      getValue() {
        return element.value;
      },
      setValue(v) {
        element.value = v;
      },
    });
    allowDragFromWidget(previewWidget);
    previewWidget.computeSize = function (width) {
      if (this.aspectRatio && !this.parentEl.hidden) {
        let height = (previewNode.size[0] - 20) / this.aspectRatio + 10;
        if (!(height > 0)) {
          height = 0;
        }
        this.computedHeight = height + 10;
        return [width, height];
      }
      return [width, -4]; //no loaded src, widget should not display
    };
    element.addEventListener(
      "contextmenu",
      (e) => {
        e.preventDefault();
        return app.canvas._mousedown_callback(e);
      },
      true
    );
    element.addEventListener(
      "pointerdown",
      (e) => {
        e.preventDefault();
        return app.canvas._mousedown_callback(e);
      },
      true
    );
    element.addEventListener(
      "mousewheel",
      (e) => {
        e.preventDefault();
        return app.canvas._mousewheel_callback(e);
      },
      true
    );
    element.addEventListener(
      "pointermove",
      (e) => {
        e.preventDefault();
        return app.canvas._mousemove_callback(e);
      },
      true
    );
    element.addEventListener(
      "pointerup",
      (e) => {
        e.preventDefault();
        return app.canvas._mouseup_callback(e);
      },
      true
    );
    previewWidget.value = {
      hidden: false,
      paused: false,
      params: {},
      muted: app.ui.settings.getSettingValue("VCW.DefaultMute"),
    };
    previewWidget.parentEl = document.createElement("div");
    // previewWidget.parentEl.className = "vcw_preview";
    previewWidget.parentEl.style["width"] = "100%";
    element.appendChild(previewWidget.parentEl);
    previewWidget.videoEl = document.createElement("video");
    previewWidget.videoEl.controls = false;
    previewWidget.videoEl.loop = true;
    previewWidget.videoEl.muted = true;
    previewWidget.videoEl.style["width"] = "100%";
    previewWidget.videoEl.addEventListener("loadedmetadata", () => {
      previewWidget.aspectRatio =
        previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
      fitHeight(this);
    });
    previewWidget.videoEl.addEventListener("error", () => {
      //TODO: consider a way to properly notify the user why a preview isn't shown.
      previewWidget.parentEl.hidden = true;
      fitHeight(this);
    });
    previewWidget.videoEl.onmouseenter = () => {
      previewWidget.videoEl.muted = previewWidget.value.muted;
    };
    previewWidget.videoEl.onmouseleave = () => {
      previewWidget.videoEl.muted = true;
    };

    previewWidget.imgEl = document.createElement("img");
    previewWidget.imgEl.style["width"] = "100%";
    previewWidget.imgEl.hidden = true;
    previewWidget.imgEl.onload = () => {
      previewWidget.aspectRatio =
        previewWidget.imgEl.naturalWidth / previewWidget.imgEl.naturalHeight;
      fitHeight(this);
    };
    previewWidget.parentEl.appendChild(previewWidget.videoEl);
    previewWidget.parentEl.appendChild(previewWidget.imgEl);
    var timeout = null;
    this.updateParameters = (params, force_update) => {
      if (!previewWidget.value.params) {
        if (typeof previewWidget.value != "object") {
          previewWidget.value = { hidden: false, paused: false };
        }
        previewWidget.value.params = {};
      }
      Object.assign(previewWidget.value.params, params);
      if (timeout) {
        clearTimeout(timeout);
      }
      if (force_update) {
        previewWidget.updateSource();
      } else {
        timeout = setTimeout(() => previewWidget.updateSource(), 100);
      }
    };
    previewWidget.updateSource = function () {
      if (this.value.params == undefined) {
        return;
      }
      let params = {};
      Object.assign(params, this.value.params); //shallow copy
      params.timestamp = Date.now();
      this.parentEl.hidden = this.value.hidden;
      if (params.format?.split("/")[0] == "video") {
        this.videoEl.autoplay = !this.value.paused && !this.value.hidden;
        this.videoEl.src = api.apiURL("/view?" + new URLSearchParams(params));
        this.videoEl.hidden = false;
        this.imgEl.hidden = true;
      }
    };
    previewWidget.callback = previewWidget.updateSource;
    previewWidget.parentEl.appendChild(previewWidget.videoEl);
    previewWidget.parentEl.appendChild(previewWidget.imgEl);
  });
}

function addLoadCommon(nodeType, nodeData) {
  addVideoPreview(nodeType);
  chainCallback(nodeType.prototype, "onNodeCreated", function () {
    for (let widget of this.widgets) {
      if (widget.type != "button") {
        widget.callback?.(widget.value);
      }
    }
  });
}

app.registerExtension({
  name: "VideoChatWrapper.Core",
  // comfyui settings
  settings: [
    {
      id: "VCW.DefaultMute",
      category: ["VideoChat", "Previews", "Default Mute"],
      name: "Mute videos by default",
      type: "boolean",
      defaultValue: false,
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData?.name == "VCW_LoadVideo") {
      chainCallback(nodeType.prototype, "onNodeCreated", function () {
        const pathWidget = this.widgets.find((w) => w.name === "video");
        chainCallback(pathWidget, "callback", (value) => {
          if (!value) {
            return;
          }
          let parts = ["input", value];
          let extension_index = parts[1].lastIndexOf(".");
          let extension = parts[1].slice(extension_index + 1);
          let format = "video";
          // if (["gif", "webp", "avif"].includes(extension)) {
          //   format = "image";
          // }
          format += "/" + extension;
          let params = { filename: parts[1], type: parts[0], format: format };
          this.updateParameters(params, true);
        });
      });
      addUploadWidget(nodeType, nodeData, "video");
      addLoadCommon(nodeType, nodeData);
    }
  },
});
