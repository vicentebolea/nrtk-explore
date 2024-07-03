r"""
Define your classes and create the instances that you need to expose
"""

import logging
import asyncio
from functools import partial
import os

from trame.ui.quasar import QLayout
from trame.widgets import quasar
from trame.widgets import html
from trame.app import get_server, asynchronous

from nrtk_explorer.library import images_manager
from nrtk_explorer.app import ui
from nrtk_explorer.app.applet import Applet
from nrtk_explorer.app.parameters import ParametersApp
from nrtk_explorer.app.image_meta import (
    update_image_meta,
    delete_image_meta,
)
import nrtk_explorer.test_data
from nrtk_explorer.app.trame_utils import delete_state
from nrtk_explorer.app.image_ids import image_id_to_dataset_id, image_id_to_result_id
from nrtk_explorer.library.dataset import get_dataset


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DIR_NAME = os.path.dirname(nrtk_explorer.test_data.__file__)
DATASET_DIRS = [
    f"{DIR_NAME}/coco-od-2017/test_val2017.json"
]


class ViewApp(Applet):
    def __init__(self, server):
        super().__init__(server)

        self.update_image_meta = partial(update_image_meta, self.server.state)

        self._parameters_app = ParametersApp(
            server=server,
        )

        self._ui = None

        self.is_standalone_app = self.server.state.parent is None
        if self.is_standalone_app:
            self.context.images_manager = images_manager.ImagesManager()

        if self.context["image_objects"] is None:
            self.context["image_objects"] = {}

        self.state.annotation_categories = {}

        self.state.source_image_ids = []

        if self.state.current_dataset is None:
            self.state.current_dataset = DATASET_DIRS[0]

        self.state.current_num_elements = 15

        self.server.controller.add("on_server_ready")(self.on_server_ready)
        self._on_hover_fn = None

    def on_server_ready(self, *args, **kwargs):
        # Bind instance methods to state change
        self.state.change("current_dataset")(self.on_current_dataset_change)
        self.state.change("current_num_elements")(self.on_current_num_elements_change)

        self.on_current_dataset_change(self.state.current_dataset)

    def on_current_num_elements_change(self, current_num_elements, **kwargs):
        dataset = get_dataset(self.state.current_dataset)
        ids = [img["id"] for img in dataset["images"]]
        return self.set_source_images(ids[:current_num_elements])

    def _update_images(self, selected_ids):
        source_image_ids = []

        current_dir = os.path.dirname(self.state.current_dataset)

        dataset = get_dataset(self.state.current_dataset)

        for selected_id in selected_ids:
            image_index = self.context.image_id_to_index[selected_id]
            if image_index >= len(dataset["images"]):
                continue

            image_metadata = dataset["images"][image_index]
            image_id = f"img_{image_metadata['id']}"
            source_image_ids.append(image_id)
            image_filename = os.path.join(current_dir, image_metadata["file_name"])
            img = self.context.images_manager.load_image(image_filename)
            self.state[image_id] = images_manager.convert_to_base64(img)
            self.context.image_objects[image_id] = img

        if len(selected_ids) > 0:
            self.state.hovered_id = ""

        old_source_image_ids = self.state.source_image_ids
        self.state.source_image_ids = source_image_ids

    async def _set_source_images(self, selected_ids):
        # We need to yield twice for the self.state.loading_images=True to
        # commit to the trame state to show a spinner
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self._update_images(selected_ids)
        with self.state:
            self.state.loading_images = False

    def set_source_images(self, selected_ids):
        if len(selected_ids):
            self.state.loading_images = True
        if hasattr(self, "_set_source_images_task"):
            self._set_source_images_task.cancel()
        self._set_source_images_task = asynchronous.create_task(
            self._set_source_images(selected_ids)
        )

    def reset_data(self):
        for image_id in self.state.source_image_ids:
            delete_state(self.state, image_id)
            if image_id in self.context["image_objects"]:
                del self.context["image_objects"][image_id]
            result_id = image_id_to_result_id(image_id)
            delete_state(self.state, result_id)

        for image_id in self.state.source_image_ids:
            dataset_id = image_id_to_dataset_id(image_id)
            delete_image_meta(self.server.state, dataset_id)

        self.state.source_image_ids = []
        self.state.annotation_categories = {}

    def on_current_dataset_change(self, current_dataset, **kwargs):
        logger.debug(f"on_current_dataset_change change {self.state}")

        self.reset_data()

        dataset = get_dataset(current_dataset)
        categories = {}

        for category in dataset["categories"]:
            categories[category["id"]] = category

        self.state.annotation_categories = categories

        self.context.image_id_to_index = {}
        for i, image in enumerate(dataset["images"]):
            self.context.image_id_to_index[image["id"]] = i

        if self.is_standalone_app:
            self.context.images_manager = images_manager.ImagesManager()

    def on_feature_extraction_model_change(self, **kwargs):
        logger.debug(f">>> on_feature_extraction_model_change change {self.state}")

        self.sync_annotations_to_state(self.state.source_image_ids)

    def sync_annotations_to_state(self, image_ids):
        for image_id in image_ids:
            result_id = image_id_to_result_id(image_id)
            self.state[result_id] = self.context["annotations"].get(image_id, [])

    def on_image_hovered(self, id):
        self.state.hovered_id = id

    def set_on_hover(self, fn):
        self._on_hover_fn = fn

    def on_hover(self, hover_event):
        id_ = hover_event["id"]
        self.on_image_hovered(id_)
        if self._on_hover_fn:
            self._on_hover_fn(id_)

    def dataset_widget(self):
        ui.ImageList(self.on_hover)

    # This is only used within when this module (file) is executed as an Standalone app.
    @property
    def ui(self):
        if self._ui is None:
            with QLayout(
                self.server, view="lhh LpR lff", classes="shadow-2 rounded-borders bg-grey-2"
            ) as layout:
                # # Toolbar
                with quasar.QHeader():
                    with quasar.QToolbar(classes="shadow-4"):
                        quasar.QBtn(
                            flat=True,
                            click="drawerLeft = !drawerLeft",
                            round=True,
                            dense=False,
                            icon="menu",
                        )
                        quasar.QToolbarTitle("View")

                # # Main content
                with quasar.QPageContainer():
                    with quasar.QPage():
                        with html.Div(classes="row"):
                            with html.Div(classes="col-2 q-pa-md"):
                                with html.Div(
                                    classes="column justify-center", style="padding:1rem"
                                ):
                                    with html.Div(classes="col"):
                                        quasar.QSelect(
                                            label="Dataset",
                                            v_model=("current_dataset",),
                                            options=(DATASET_DIRS,),
                                            filled=True,
                                            emit_value=True,
                                            map_options=True,
                                        )

                                        html.P("Number of elements:", classes="text-body2")
                                        quasar.QSlider(
                                            v_model=("current_num_elements",),
                                            min=(0,),
                                            max=(25,),
                                            step=(1,),
                                            label=True,
                                            label_always=True,
                                        )

                            self.dataset_widget()

                self._ui = layout
        return self._ui


def view(server=None, *args, **kwargs):
    server = get_server()
    server.client_type = "vue3"

    transforms_app = ViewApp(server)
    transforms_app.ui

    server.start(**kwargs)


if __name__ == "__main__":
    transforms()
