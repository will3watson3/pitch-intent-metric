import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import labeler_server


class FramePayloadTests(unittest.TestCase):
    def test_target_audit_returns_filtered_frames_and_saved_reviews(self):
        frames = [dict(pitch_uid='p', frame_uid='p_1', sample_pitcher_name='Dylan Cease',
                       pitch_type='SL', setup_visible='yes', target_confidence_1_to_5='4'),
                  dict(pitch_uid='other', frame_uid='other_1', sample_pitcher_name='Other', pitch_type='FF')]
        audit = [dict(pitch_uid='p', frame_uid='p_1', audit_status='confirmed')]
        with patch.object(labeler_server, 'frame_review_payload', return_value=dict(frames=frames, calibrations={})), \
             patch.object(labeler_server, 'latest_analysis_file', return_value=None), \
             patch.object(labeler_server, 'read_target_audit_labels', return_value=audit):
            result = labeler_server.target_audit_payload()
        self.assertEqual(result['pitch_count'], 1)
        self.assertEqual(result['usable_pitch_count'], 1)
        self.assertEqual(result['reviewed_pitch_count'], 1)
        self.assertEqual(result['frames'][0]['saved_audit_status'], 'confirmed')

    def test_empty_target_audit_returns_an_empty_queue(self):
        with patch.object(labeler_server, 'frame_review_payload', return_value=dict(frames=[], calibrations={})), \
             patch.object(labeler_server, 'latest_analysis_file', return_value=None), \
             patch.object(labeler_server, 'read_target_audit_labels', return_value=[]):
            result = labeler_server.target_audit_payload()
        self.assertEqual(result['pitch_count'], 0)
        self.assertEqual(result['frames'], [])

    def test_frames_do_not_inherit_selected_pitch_geometry(self):
        frames = [dict(pitch_uid='p', frame_uid='p_1', image_path=''), dict(pitch_uid='p', frame_uid='p_2', image_path='')]
        selected = {'p': dict(frame_uid='p_1', zone_x='640', zone_y='300', zone_width='43', zone_height='55', vision_target_x_01='.2')}
        per_frame = {'p_2': dict(zone_x='652', zone_y='310', zone_width='45', zone_height='60', zone_refinement_status='visible_four_edge_fit')}
        def index(path, key):
            if key == 'frame_uid': return per_frame
            if path == labeler_server.VISION_REVIEW_LABELS_PATH: return {}
            return selected
        with patch.object(labeler_server, 'read_csv_records', return_value=frames), patch.object(labeler_server, 'indexed_records', side_effect=index), patch.object(labeler_server, 'latest_analysis_file', return_value=None):
            result = labeler_server.frame_review_payload()['frames']
        self.assertEqual(result[0]['zone_x'], '640')
        self.assertEqual(result[1]['zone_x'], '652')
        self.assertEqual(result[1]['vision_target_x_01'], '')

    def test_small_manual_zones_keep_exact_size(self):
        overlay = labeler_server.normalize_overlay(dict(left=48, top=35, width=3.2, height=5.5))
        self.assertEqual(overlay['width'], 3.2)
        self.assertEqual(overlay['height'], 5.5)


if __name__ == '__main__':
    unittest.main()
