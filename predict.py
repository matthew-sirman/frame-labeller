import os
import delphin.codecs.eds as eds

import config as cfg
import train
import loader


def create_resource_from_semlinks(semlinks, model, write_to):
    linked = semlinks.eds_dataset
    if not hasattr(semlinks, "eds_unlinked"):
        raise Exception("Make sure GENERATE_RESOURCE=True in config")
    unlinked = semlinks.eds_unlinked

    entries = []

    for deepbank, _, (frame_root, frame), roles in linked:
        entry = f"{deepbank} {frame_root} {frame}"
        for (_, node), role in roles.items():
            entry += f" {frame_root}->{node} {role}"

        entries.append(entry)

    step = 0

    for deepbank, eds_data in unlinked:
        try:
            eds_data = eds.decode(eds_data)
        except eds.EDSSyntaxError:
            continue
        step += 1
        (frame_root, frame), roles = model.predict(eds_data)
        entry = f"{deepbank} {frame_root} {frame}"
        for _, node, role in roles:
            entry += f" {frame_root}->{node} {role}"

        entries.append(entry)

        if step % cfg.LOG_PREDS_AFTER_N == 0:
            print(f"Completed {step}/{len(unlinked)} predictions.")

    with open(write_to, "w") as out_f:
        for entry in entries:
            out_f.write(entry + os.linesep)


def create_resource():
    print("Loading semlinks...")
    semlinks = loader.construct_semlink_dataset_from_config()
    print("Loaded")
    model = train.load_model_from_cfg(semlinks)

    if cfg.LOAD_MODEL is not None:
        train.load_model_params(model)
    else:
        raise Exception("Make sure model load path is specified for "
                        "GENERATE_RESOURCE=True mode")

    create_resource_from_semlinks(semlinks, model, cfg.WRITE_RESOURCE)
