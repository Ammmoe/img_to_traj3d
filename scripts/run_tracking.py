import os
import csv
import cv2

from img_to_traj3d.io.load_images import load_image_sequence, load_image
from img_to_traj3d.tracking.kcf_tracker import KCFTracker
from img_to_traj3d.tracking.bbox_utils import bbox_is_valid


def ensure_dir(folder_path):
    os.makedirs(folder_path, exist_ok=True)


def run_tracking(
    raw_images_folder="data/raw_images",
    output_csv_path="data/outputs/csv/tracked_results.csv",
    output_images_folder="data/outputs/images/tracking_vis",
    save_visualization=True,
    show_window=True
):
    ensure_dir(os.path.dirname(output_csv_path))

    if save_visualization:
        ensure_dir(output_images_folder)

    image_paths = load_image_sequence(raw_images_folder, extensions=(".png",))
    if len(image_paths) == 0:
        raise RuntimeError(f"No PNG images found in: {raw_images_folder}")

    # Load first frame
    first_frame = load_image(image_paths[0])

    print("Draw initial bounding box on the drone and press ENTER/SPACE.")
    bbox = cv2.selectROI("Select ROI", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    if not bbox_is_valid(bbox):
        raise RuntimeError("Invalid ROI selected. Tracking aborted.")

    tracker = KCFTracker()
    tracker.initialize(first_frame, bbox)

    results = []

    for frame_idx, img_path in enumerate(image_paths):
        frame = load_image(img_path)
        image_name = os.path.basename(img_path)

        bbox, center, success = tracker.update(frame)

        if success:
            x, y, w, h = bbox
            u, v = center
        else:
            x = y = w = h = -1
            u = v = -1

        results.append({
            "frame": frame_idx,
            "image": image_name,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "u": u,
            "v": v,
            "success": int(success)
        })

        # Visualization
        if show_window or save_visualization:
            vis = frame.copy()

            if success:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)

            cv2.putText(
                vis,
                f"Frame {frame_idx} | Success: {success}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            if save_visualization:
                out_img_path = os.path.join(output_images_folder, image_name)
                cv2.imwrite(out_img_path, vis)

            if show_window:
                cv2.imshow("KCF Tracking", vis)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    print("Stopped by user.")
                    break

    cv2.destroyAllWindows()

    # Save CSV output
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame", "image", "x", "y", "w", "h", "u", "v", "success"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"[DONE] Tracking results saved to: {output_csv_path}")
    print(f"[DONE] Visualization images saved to: {output_images_folder}")


if __name__ == "__main__":
    run_tracking()
