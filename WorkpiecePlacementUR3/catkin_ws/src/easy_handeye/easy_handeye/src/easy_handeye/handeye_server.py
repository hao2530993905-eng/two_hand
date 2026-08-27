import itertools
import json
import os

import easy_handeye_msgs as ehm
import rospy
import std_msgs
import std_srvs
from easy_handeye_msgs import msg, srv
from std_msgs import msg
from std_srvs import srv

import easy_handeye as hec
from easy_handeye.handeye_calibration import HandeyeCalibration, HandeyeCalibrationParameters
from easy_handeye.handeye_calibration_backend_opencv import HandeyeCalibrationBackendOpenCV
from easy_handeye.handeye_sampler import HandeyeSampler
from geometry_msgs.msg import TransformStamped


class HandeyeServer:
    def __init__(self, namespace=None):
        if namespace is None:
            namespace = rospy.get_namespace()
        self.parameters = HandeyeCalibrationParameters.init_from_parameter_server(namespace)

        self.sampler = HandeyeSampler(handeye_parameters=self.parameters)
        self._restore_samples_from_environment()
        self.calibration_backends = {'OpenCV': HandeyeCalibrationBackendOpenCV()}
        self.calibration_algorithm = 'OpenCV/Tsai-Lenz'

        # setup calibration services and topics

        self.list_algorithms_service = rospy.Service(hec.LIST_ALGORITHMS_TOPIC,
                                                     ehm.srv.ListAlgorithms, self.list_algorithms)
        self.set_algorithm_service = rospy.Service(hec.SET_ALGORITHM_TOPIC, ehm.srv.SetAlgorithm, self.set_algorithm)
        self.get_sample_list_service = rospy.Service(hec.GET_SAMPLE_LIST_TOPIC,
                                                     ehm.srv.TakeSample, self.get_sample_lists)
        self.take_sample_service = rospy.Service(hec.TAKE_SAMPLE_TOPIC,
                                                 ehm.srv.TakeSample, self.take_sample)
        self.remove_sample_service = rospy.Service(hec.REMOVE_SAMPLE_TOPIC,
                                                   ehm.srv.RemoveSample, self.remove_sample)
        self.compute_calibration_service = rospy.Service(hec.COMPUTE_CALIBRATION_TOPIC,
                                                         ehm.srv.ComputeCalibration, self.compute_calibration)
        self.save_calibration_service = rospy.Service(hec.SAVE_CALIBRATION_TOPIC,
                                                      std_srvs.srv.Empty, self.save_calibration)

        # Useful for secondary input sources (e.g. programmable buttons on robot)
        self.take_sample_topic = rospy.Subscriber(hec.TAKE_SAMPLE_TOPIC,
                                                  std_msgs.msg.Empty, self.take_sample)
        # Do not also subscribe to REMOVE_SAMPLE_TOPIC here.  The rqt GUI uses
        # the service with the same resolved name; keeping an Empty subscriber
        # on it can invoke remove_last_sample in addition to the requested
        # indexed deletion, so one click may remove two sample pairs.

        self.last_calibration = None

    @staticmethod
    def _transform_from_backup(values, parent_frame, child_frame):
        translation = values['translation_xyz_m']
        quaternion = values['quaternion_xyzw']
        if len(translation) != 3 or len(quaternion) != 4:
            raise ValueError('invalid transform dimensions in sample backup')
        message = TransformStamped()
        message.header.stamp = rospy.Time(0)
        message.header.frame_id = parent_frame
        message.child_frame_id = child_frame
        message.transform.translation.x = float(translation[0])
        message.transform.translation.y = float(translation[1])
        message.transform.translation.z = float(translation[2])
        message.transform.rotation.x = float(quaternion[0])
        message.transform.rotation.y = float(quaternion[1])
        message.transform.rotation.z = float(quaternion[2])
        message.transform.rotation.w = float(quaternion[3])
        return message

    def _restore_samples_from_environment(self):
        """Restore evaluator raw_samples when explicitly requested at launch."""
        filename = os.environ.get('EASY_HANDEYE_SAMPLE_BACKUP', '').strip()
        if not filename:
            return
        filename = os.path.abspath(os.path.expanduser(filename))
        with open(filename, 'r') as stream:
            document = json.load(stream)
        raw_samples = document.get('raw_samples')
        if not isinstance(raw_samples, list):
            raise ValueError('{} does not contain raw_samples'.format(filename))
        if document.get('namespace', '').strip('/') != self.parameters.namespace.strip('/'):
            raise ValueError(
                'sample backup namespace {} does not match {}'.format(
                    document.get('namespace'), self.parameters.namespace
                )
            )

        restored = []
        for sample in raw_samples:
            robot = self._transform_from_backup(
                sample['tool0_from_base_link'],
                self.parameters.robot_effector_frame,
                self.parameters.robot_base_frame,
            )
            optical = self._transform_from_backup(
                sample['camera_link_from_marker'],
                self.parameters.tracking_base_frame,
                self.parameters.tracking_marker_frame,
            )
            restored.append({'robot': robot, 'optical': optical})
        self.sampler.samples = restored
        rospy.logwarn(
            'Restored %d hand-eye sample pairs from %s', len(restored), filename
        )

    # algorithm

    def list_algorithms(self, _):
        algorithms_nested = [[bck_name + '/' + alg_name for alg_name in bck.AVAILABLE_ALGORITHMS] for bck_name, bck in
                             self.calibration_backends.items()]
        available_algorithms = list(itertools.chain(*algorithms_nested))
        return ehm.srv.ListAlgorithmsResponse(algorithms=available_algorithms,
                                              current_algorithm=self.calibration_algorithm)

    def set_algorithm(self, req):
        alg_to_set = req.new_algorithm
        bckname_algname = alg_to_set.split('/')
        if len(bckname_algname) != 2:
            return ehm.srv.SetAlgorithmResponse(success=False)
        bckname, algname = bckname_algname
        if bckname not in self.calibration_backends:
            return ehm.srv.SetAlgorithmResponse(success=False)
        if algname not in self.calibration_backends[bckname].AVAILABLE_ALGORITHMS:
            return ehm.srv.SetAlgorithmResponse(success=False)
        rospy.loginfo('switching to calibration algorithm {}'.format(alg_to_set))
        self.calibration_algorithm = alg_to_set
        return ehm.srv.SetAlgorithmResponse(success=True)

    # sampling

    def _retrieve_sample_list(self):
        ret = ehm.msg.SampleList()
        for s in self.sampler.get_samples():
            ret.camera_marker_samples.append(s['optical'].transform)
            ret.hand_world_samples.append(s['robot'].transform)
        return ret

    def get_sample_lists(self, _):
        return ehm.srv.TakeSampleResponse(self._retrieve_sample_list())

    def take_sample(self, _):
        self.sampler.take_sample()
        return ehm.srv.TakeSampleResponse(self._retrieve_sample_list())

    def remove_last_sample(self):
        self.sampler.remove_sample(len(self.sampler.samples) - 1)

    def remove_sample(self, req):
        try:
            self.sampler.remove_sample(req.sample_index)
        except IndexError:
            rospy.logerr('Invalid index ' + req.sample_index)
        return ehm.srv.RemoveSampleResponse(self._retrieve_sample_list())

    # calibration

    def compute_calibration(self, _):
        samples = self.sampler.get_samples()

        bckname, algname = self.calibration_algorithm.split('/')
        backend = self.calibration_backends[bckname]

        self.last_calibration = backend.compute_calibration(self.parameters, samples, algorithm=algname)
        ret = ehm.srv.ComputeCalibrationResponse()
        if self.last_calibration is None:
            rospy.logwarn('No valid calibration computed')
            ret.valid = False
            return ret
        ret.valid = True
        ret.calibration.eye_on_hand = self.last_calibration.parameters.eye_on_hand
        ret.calibration.transform = self.last_calibration.transformation
        return ret

    def save_calibration(self, _):
        if self.last_calibration:
            HandeyeCalibration.to_file(self.last_calibration)
            rospy.loginfo('Calibration saved to {}'.format(self.last_calibration.filename()))
        return std_srvs.srv.EmptyResponse()

    # TODO: evaluation
