extends Node3D
## Controls the 3D pirate avatar including animations, expressions, and lip-sync.
##
## This script manages:
## - Idle animations
## - Expression blend shapes
## - Lip-sync with viseme data
## - Gaze direction
## - Audio playback

# Node references (set in editor or via code)
@export var skeleton: Skeleton3D
@export var mesh: MeshInstance3D
@export var animation_player: AnimationPlayer
@export var audio_player: AudioStreamPlayer3D

# Head bone for gaze tracking
@export var head_bone_name: String = "Head"

# Blend shape names for expressions
const EXPRESSION_SHAPES = {
	"neutral": {},
	"happy": {"smile": 1.0, "eyebrow_raise": 0.3},
	"surprised": {"mouth_open": 0.5, "eyebrow_raise": 0.8},
	"angry": {"frown": 0.8, "eyebrow_lower": 0.5},
	"sad": {"frown": 0.5, "eyebrow_sad": 0.6},
	"laugh": {"smile": 1.0, "mouth_open": 0.4, "eyebrow_raise": 0.2},
	"thinking": {"eyebrow_raise": 0.3, "look_up": 0.5},
}

# Viseme blend shape mapping (Rhubarb output -> blend shape name)
const VISEME_SHAPES = {
	"A": "viseme_aa",      # "ah" - jaw open
	"B": "viseme_rest",    # closed mouth consonants (m, b, p)
	"C": "viseme_E",       # "eh" sounds
	"D": "viseme_aa",      # "ah" variant
	"E": "viseme_E",       # "ee" sounds
	"F": "viseme_U",       # "f" and "v"
	"G": "viseme_O",       # "oh" sounds
	"H": "viseme_rest",    # rest/neutral
	"X": "viseme_rest",    # silence
}

# State
var _current_expression: String = "neutral"
var _current_animation: String = "idle"
var _is_talking: bool = false
var _gaze_target: Vector3 = Vector3(0, 0, 1)
var _head_bone_idx: int = -1

# Viseme playback
var _visemes: Array = []
var _viseme_index: int = 0
var _audio_start_time: float = 0.0


func _ready() -> void:
	# Register with WebSocket server
	if WebSocketServer:
		WebSocketServer.pirate_controller = self

	# Find head bone for gaze tracking
	if skeleton:
		_head_bone_idx = skeleton.find_bone(head_bone_name)

	# Start idle animation
	_play_idle_animation()


func _process(delta: float) -> void:
	# Update gaze
	_update_gaze(delta)

	# Update lip-sync if audio is playing
	if _is_talking and audio_player and audio_player.playing:
		_update_lipsync()


func _play_idle_animation() -> void:
	"""Play random idle animation."""
	if animation_player:
		var idle_anims = ["idle", "idle_breathing", "idle_look_around"]
		var chosen = idle_anims[randi() % idle_anims.size()]

		if animation_player.has_animation(chosen):
			animation_player.play(chosen)
			_current_animation = chosen


func play_audio_with_visemes(audio_path: String, visemes) -> void:
	"""Play audio file with lip-sync viseme data."""
	print("Playing audio: %s" % audio_path)

	# Load audio file
	var audio_stream = _load_audio_file(audio_path)
	if not audio_stream:
		push_error("Failed to load audio: %s" % audio_path)
		return

	# Store visemes for lip-sync
	_visemes = visemes if visemes else []
	_viseme_index = 0
	_is_talking = true

	# Play audio
	audio_player.stream = audio_stream
	audio_player.play()
	_audio_start_time = Time.get_ticks_msec() / 1000.0

	# Start talking animation if available
	if animation_player and animation_player.has_animation("talking"):
		animation_player.play("talking")


func _load_audio_file(path: String) -> AudioStream:
	"""Load audio file from absolute path."""
	# For external files, we need to load them dynamically
	var file = FileAccess.open(path, FileAccess.READ)
	if not file:
		return null

	var buffer = file.get_buffer(file.get_length())
	file.close()

	# Create AudioStreamWAV from buffer
	var stream = AudioStreamWAV.new()
	stream.data = buffer
	stream.format = AudioStreamWAV.FORMAT_16_BITS
	stream.stereo = false
	stream.mix_rate = 24000  # Kokoro default

	return stream


func _update_lipsync() -> void:
	"""Update mouth blend shapes based on viseme timing."""
	if _visemes.is_empty():
		# Fallback: amplitude-based lip-sync
		_amplitude_lipsync()
		return

	var current_time = (Time.get_ticks_msec() / 1000.0) - _audio_start_time

	# Find current viseme
	while _viseme_index < _visemes.size():
		var viseme = _visemes[_viseme_index]
		var start_time = viseme.get("start", 0.0)
		var end_time = viseme.get("end", 0.0)

		if current_time >= start_time and current_time < end_time:
			# Apply this viseme
			var shape_name = viseme.get("shape", "X")
			_apply_viseme(shape_name)
			break
		elif current_time >= end_time:
			_viseme_index += 1
		else:
			break

	# Check if audio finished
	if not audio_player.playing:
		_on_audio_finished()


func _amplitude_lipsync() -> void:
	"""Simple amplitude-based mouth movement."""
	if not mesh:
		return

	# Get audio level (this is a simplification)
	# In real implementation, you'd analyze the audio buffer
	var mouth_open = 0.0

	if audio_player and audio_player.playing:
		# Oscillate mouth for talking effect
		mouth_open = abs(sin(Time.get_ticks_msec() * 0.01)) * 0.5

	_set_blend_shape("viseme_aa", mouth_open)


func _apply_viseme(shape_code: String) -> void:
	"""Apply a viseme blend shape."""
	if not mesh:
		return

	var blend_shape_name = VISEME_SHAPES.get(shape_code, "viseme_rest")

	# Reset all viseme shapes
	for viseme_shape in VISEME_SHAPES.values():
		_set_blend_shape(viseme_shape, 0.0)

	# Apply current viseme
	_set_blend_shape(blend_shape_name, 1.0)


func _on_audio_finished() -> void:
	"""Called when audio playback finishes."""
	_is_talking = false
	_visemes = []
	_viseme_index = 0

	# Reset mouth
	_apply_viseme("X")

	# Return to idle animation
	_play_idle_animation()

	print("Audio finished")


func stop_audio() -> void:
	"""Stop current audio playback."""
	if audio_player:
		audio_player.stop()
	_on_audio_finished()


func set_expression(expression_name: String) -> void:
	"""Set facial expression using blend shapes."""
	print("Setting expression: %s" % expression_name)

	if not mesh:
		return

	_current_expression = expression_name

	# Reset all expression shapes
	for expr in EXPRESSION_SHAPES.values():
		for shape_name in expr.keys():
			_set_blend_shape(shape_name, 0.0)

	# Apply new expression
	var shapes = EXPRESSION_SHAPES.get(expression_name, {})
	for shape_name in shapes.keys():
		_set_blend_shape(shape_name, shapes[shape_name])


func _set_blend_shape(name: String, value: float) -> void:
	"""Set a blend shape value on the mesh."""
	if not mesh:
		return

	var idx = mesh.find_blend_shape_by_name(name)
	if idx >= 0:
		mesh.set_blend_shape_value(idx, value)


func play_animation(animation_name: String, loop: bool = false) -> void:
	"""Play a specific animation."""
	print("Playing animation: %s (loop=%s)" % [animation_name, loop])

	if not animation_player:
		return

	if animation_player.has_animation(animation_name):
		animation_player.play(animation_name)
		_current_animation = animation_name

		# Set loop mode
		var anim = animation_player.get_animation(animation_name)
		if anim:
			anim.loop_mode = Animation.LOOP_LINEAR if loop else Animation.LOOP_NONE


func set_gaze(target: Vector3) -> void:
	"""Set gaze target (normalized screen coordinates)."""
	# Convert from screen coords (0-1, 0-1) to 3D direction
	# Assuming camera looking at -Z
	_gaze_target = Vector3(
		(target.x - 0.5) * 2.0,  # -1 to 1 horizontal
		(0.5 - target.y) * 2.0,  # -1 to 1 vertical (inverted)
		target.z                  # depth
	)


func _update_gaze(delta: float) -> void:
	"""Update head bone rotation to look at target."""
	if _head_bone_idx < 0 or not skeleton:
		return

	# Calculate target rotation
	var target_dir = _gaze_target.normalized()
	var head_rotation = Quaternion.IDENTITY

	# Limit rotation range for natural look
	var max_angle = deg_to_rad(30)
	var yaw = clamp(atan2(target_dir.x, target_dir.z), -max_angle, max_angle)
	var pitch = clamp(asin(target_dir.y), -max_angle * 0.7, max_angle * 0.7)

	head_rotation = Quaternion.from_euler(Vector3(pitch, yaw, 0))

	# Smoothly interpolate
	var current = skeleton.get_bone_pose_rotation(_head_bone_idx)
	var new_rotation = current.slerp(head_rotation, delta * 5.0)
	skeleton.set_bone_pose_rotation(_head_bone_idx, new_rotation)


func reset_to_default() -> void:
	"""Reset avatar to default state."""
	stop_audio()
	set_expression("neutral")
	_gaze_target = Vector3(0, 0, 1)
	_play_idle_animation()
