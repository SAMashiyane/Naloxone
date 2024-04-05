# Author: Salio(Mohtarami)
# https://github.com/SAMashiyane
# Date: 2023-2024

import inspect
from typing import Any, Dict, Optional
import naloxlib.efficacy.depend

class BaseContainer:

    def __init__(
        self,
        id: str,
        name: str,
        class_def: type,
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not args:
            args = {}
        if not isinstance(args, dict):
            raise TypeError("args needs to be a dictionary.")

        self.id = id
        self.name = name
        self.class_def = class_def
        self.reference = self.get_class_name()
        self.args = args
        self.active = True

    def get_class_name(self):
        return naloxlib.efficacy.depend.get_class_name(self.class_def)

    def get_package_name(self):
        return naloxlib.efficacy.depend.get_package_name(self.class_def)

    def get_dict(self, internal: bool = True) -> Dict[str, Any]:
        d = [("ID", self.id), ("Name", self.name), ("Reference", self.reference)]

        if internal:
            d += [
                ("Class", self.class_def),
                ("Args", self.args),
            ]

        return dict(d)

def get_all_containers(
    container_globals: dict,
    experiment: Any,
    type_var: type,
    raise_errors: bool = True,
) -> Dict[str, BaseContainer]:
    model_container_classes = [
        obj
        for _, obj in container_globals.items()
        if inspect.isclass(obj)
        and type_var in tuple(x for x in inspect.getmro(obj) if x != obj)
    ]

    model_containers = []

    for obj in model_container_classes:
        if raise_errors:
            if hasattr(obj, "active") and not obj.active:
                continue
            instance = obj(experiment)
            if instance.active:
                model_containers.append(instance)
        else:
            try:
                if hasattr(obj, "active") and not obj.active:
                    continue
                instance = obj(experiment)
                if instance.active:
                    model_containers.append(instance)
            except Exception:
                pass

    return {container.id: container for container in model_containers}
