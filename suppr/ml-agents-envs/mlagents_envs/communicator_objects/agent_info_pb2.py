# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mlagents_envs/communicator_objects/agent_info.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mlagents_envs.communicator_objects import observation_pb2 as mlagents__envs_dot_communicator__objects_dot_observation__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='mlagents_envs/communicator_objects/agent_info.proto',
  package='communicator_objects',
  syntax='proto3',
  serialized_pb=_b('\n3mlagents_envs/communicator_objects/agent_info.proto\x12\x14\x63ommunicator_objects\x1a\x34mlagents_envs/communicator_objects/observation.proto\"\xd1\x01\n\x0e\x41gentInfoProto\x12\x0e\n\x06reward\x18\x07 \x01(\x02\x12\x0c\n\x04\x64one\x18\x08 \x01(\x08\x12\x18\n\x10max_step_reached\x18\t \x01(\x08\x12\n\n\x02id\x18\n \x01(\x05\x12\x13\n\x0b\x61\x63tion_mask\x18\x0b \x03(\x08\x12<\n\x0cobservations\x18\r \x03(\x0b\x32&.communicator_objects.ObservationProtoJ\x04\x08\x01\x10\x02J\x04\x08\x02\x10\x03J\x04\x08\x03\x10\x04J\x04\x08\x04\x10\x05J\x04\x08\x05\x10\x06J\x04\x08\x06\x10\x07J\x04\x08\x0c\x10\rB\x1f\xaa\x02\x1cMLAgents.CommunicatorObjectsb\x06proto3')
  ,
  dependencies=[mlagents__envs_dot_communicator__objects_dot_observation__pb2.DESCRIPTOR,])




_AGENTINFOPROTO = _descriptor.Descriptor(
  name='AgentInfoProto',
  full_name='communicator_objects.AgentInfoProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='reward', full_name='communicator_objects.AgentInfoProto.reward', index=0,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='done', full_name='communicator_objects.AgentInfoProto.done', index=1,
      number=8, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_step_reached', full_name='communicator_objects.AgentInfoProto.max_step_reached', index=2,
      number=9, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='id', full_name='communicator_objects.AgentInfoProto.id', index=3,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='action_mask', full_name='communicator_objects.AgentInfoProto.action_mask', index=4,
      number=11, type=8, cpp_type=7, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='observations', full_name='communicator_objects.AgentInfoProto.observations', index=5,
      number=13, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=132,
  serialized_end=341,
)

_AGENTINFOPROTO.fields_by_name['observations'].message_type = mlagents__envs_dot_communicator__objects_dot_observation__pb2._OBSERVATIONPROTO
DESCRIPTOR.message_types_by_name['AgentInfoProto'] = _AGENTINFOPROTO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

AgentInfoProto = _reflection.GeneratedProtocolMessageType('AgentInfoProto', (_message.Message,), dict(
  DESCRIPTOR = _AGENTINFOPROTO,
  __module__ = 'mlagents_envs.communicator_objects.agent_info_pb2'
  # @@protoc_insertion_point(class_scope:communicator_objects.AgentInfoProto)
  ))
_sym_db.RegisterMessage(AgentInfoProto)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\252\002\034MLAgents.CommunicatorObjects'))
# @@protoc_insertion_point(module_scope)
