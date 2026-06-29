import numpy as np
from datasets import Dataset
from PIL import Image

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.motionatlas_bench import utils as motionatlas_utils


def _mcq(sample_id, idx, answer_index=0):
    return {
        "id": f"sample_{sample_id}_mcq_{idx}",
        "sample_id": sample_id,
        "event_id": 1,
        "video_path": f"videos/sample_{sample_id}.mp4",
        "video_type": "video",
        "target_entity": {"name": "target", "visual_id": "green object"},
        "question": f"question {idx}",
        "options": ["target moves left", "The description does not mention the target motion.", "The motion value differs from all listed options."],
        "answer": "target moves left",
        "answer_index": answer_index,
    }


def _encode_uncompressed_rle(mask):
    flat = mask.reshape(-1, order="F")
    counts = []
    current = 0
    run = 0
    for value in flat:
        if int(value) == current:
            run += 1
        else:
            counts.append(run)
            current = int(value)
            run = 1
    counts.append(run)
    return {"size": list(mask.shape), "counts": counts}


def test_motionatlas_task_is_registered():
    task_manager = TaskManager()
    assert "motionatlas_bench_first" in task_manager.all_tasks


def test_motionatlas_process_docs_groups_mcqs_by_sample():
    dataset = Dataset.from_list([_mcq(0, 1), _mcq(0, 2), _mcq(1, 1)])
    processed = motionatlas_utils.motionatlas_process_docs(dataset)

    assert len(processed) == 2
    first = processed[0]
    assert first["sample_id"] == 0
    assert len(first["mcqs"]) == 2
    assert first["video_path"] == "videos/sample_0.mp4"


def test_frame_reader_sorts_image_directory(tmp_path):
    for name, value in [("00010.jpg", 10), ("00002.jpg", 2), ("00001.jpg", 1)]:
        image = Image.fromarray(np.full((3, 3, 3), value, dtype=np.uint8))
        image.save(tmp_path / name)

    reader = motionatlas_utils.FrameReader(tmp_path)
    try:
        assert [path.name for path in reader.image_paths] == ["00001.jpg", "00002.jpg", "00010.jpg"]
    finally:
        reader.close()


def test_uncompressed_rle_decode_and_contour_render():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    rle = _encode_uncompressed_rle(mask)

    decoded = motionatlas_utils.decode_rle_mask(rle)
    assert np.array_equal(decoded, mask)

    frame = np.zeros((6, 6, 3), dtype=np.uint8)
    rendered = motionatlas_utils.draw_mask_contour(frame, rle, thickness=1)
    assert rendered[:, :, 1].max() == 255


def test_caption_prompt_is_caption_oriented():
    prompt = motionatlas_utils.build_caption_prompt()
    assert "detailed motion caption" in prompt
    assert "Do not answer any multiple-choice question." in prompt
    assert "Return JSON" not in prompt


def test_process_results_with_fake_judge(monkeypatch):
    class FakeJudge:
        model = "fake-gemini"

        def generate(self, prompt, max_tokens, temperature, top_p):
            if "question 1" in prompt:
                return '{"answer": "A"}'
            return '{"answer": "B"}'

    monkeypatch.setattr(motionatlas_utils, "create_judge_client", lambda: FakeJudge())

    doc = {
        "sample_id": 0,
        "video_path": "videos/sample_0.mp4",
        "video_type": "video",
        "target_entity": {"name": "target"},
        "mcqs": [_mcq(0, 1), _mcq(0, 2)],
    }
    processed = motionatlas_utils.motionatlas_process_results(doc, ["The target moves left across the scene."])
    payload = processed["motionatlas_weighted_score"]
    results = [payload]

    assert len(payload["mcq_results"]) == 2
    assert motionatlas_utils.motionatlas_aggregate_accuracy(results) == 0.5
    assert motionatlas_utils.motionatlas_aggregate_weighted_score(results) == 0.5
    assert motionatlas_utils.motionatlas_aggregate_recall(results) == 0.5
    assert motionatlas_utils.motionatlas_aggregate_precision(results) == 1.0
    assert motionatlas_utils.motionatlas_aggregate_answered_rate(results) == 1.0
