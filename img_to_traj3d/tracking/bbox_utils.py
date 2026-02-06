def bbox_to_int(bbox):
    x, y, w, h = bbox
    return int(x), int(y), int(w), int(h)


def bbox_center(bbox):
    x, y, w, h = bbox
    u = x + w / 2.0
    v = y + h / 2.0
    return u, v


def bbox_is_valid(bbox):
    _, _, w, h = bbox
    return (w > 0 and h > 0)
