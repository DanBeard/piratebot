extends Node
## WebSocket server for receiving commands from Python PirateBot.
##
## This autoload script runs a WebSocket server that accepts connections
## from the Python orchestrator and routes commands to the avatar controller.

const PORT = 9876

var _server: TCPServer
var _clients: Array[WebSocketPeer] = []
var _pending_peers: Array[StreamPeerTCP] = []

# Reference to the pirate controller (set by the main scene)
var pirate_controller: Node = null

signal command_received(command: Dictionary)


func _ready() -> void:
	_server = TCPServer.new()
	var err = _server.listen(PORT)
	if err != OK:
		push_error("Failed to start WebSocket server on port %d: %s" % [PORT, err])
		return
	print("WebSocket server listening on port %d" % PORT)


func _process(_delta: float) -> void:
	# Check for new connections
	if _server.is_connection_available():
		var peer = _server.take_connection()
		if peer:
			_pending_peers.append(peer)
			print("New TCP connection pending")

	# Process pending peers (WebSocket handshake)
	var i = _pending_peers.size() - 1
	while i >= 0:
		var peer = _pending_peers[i]
		peer.poll()

		if peer.get_status() == StreamPeerTCP.STATUS_CONNECTED:
			# Try to upgrade to WebSocket
			var ws = WebSocketPeer.new()
			var err = ws.accept_stream(peer)
			if err == OK:
				_clients.append(ws)
				_pending_peers.remove_at(i)
				print("Client connected via WebSocket")
			else:
				# Not ready yet or failed, keep trying
				pass
		elif peer.get_status() == StreamPeerTCP.STATUS_ERROR:
			_pending_peers.remove_at(i)

		i -= 1

	# Process connected WebSocket clients
	i = _clients.size() - 1
	while i >= 0:
		var ws = _clients[i]
		ws.poll()

		var state = ws.get_ready_state()

		if state == WebSocketPeer.STATE_OPEN:
			while ws.get_available_packet_count() > 0:
				var packet = ws.get_packet()
				_handle_message(ws, packet.get_string_from_utf8())
		elif state == WebSocketPeer.STATE_CLOSING:
			# Keep polling until closed
			pass
		elif state == WebSocketPeer.STATE_CLOSED:
			var code = ws.get_close_code()
			var reason = ws.get_close_reason()
			print("Client disconnected: %d - %s" % [code, reason])
			_clients.remove_at(i)

		i -= 1


func _handle_message(client: WebSocketPeer, message: String) -> void:
	"""Handle incoming JSON command from Python."""
	var json = JSON.new()
	var err = json.parse(message)

	if err != OK:
		push_warning("Invalid JSON received: %s" % message)
		_send_error(client, -1, "Invalid JSON")
		return

	var data = json.data
	if not data is Dictionary:
		_send_error(client, -1, "Expected JSON object")
		return

	var msg_id = data.get("id", -1)
	var cmd_type = data.get("type", "")

	print("Received command: %s (id=%d)" % [cmd_type, msg_id])

	# Handle different command types
	match cmd_type:
		"handshake":
			_handle_handshake(client, msg_id, data)
		"play_audio":
			_handle_play_audio(client, msg_id, data)
		"stop_audio":
			_handle_stop_audio(client, msg_id)
		"set_expression":
			_handle_set_expression(client, msg_id, data)
		"play_animation":
			_handle_play_animation(client, msg_id, data)
		"set_gaze":
			_handle_set_gaze(client, msg_id, data)
		"reset":
			_handle_reset(client, msg_id)
		_:
			_send_error(client, msg_id, "Unknown command: %s" % cmd_type)

	# Emit signal for any listeners
	command_received.emit(data)


func _handle_handshake(client: WebSocketPeer, msg_id: int, data: Dictionary) -> void:
	var client_name = data.get("client", "unknown")
	var version = data.get("version", "unknown")
	print("Handshake from %s v%s" % [client_name, version])
	_send_response(client, msg_id, {"status": "ok", "server": "PirateBot Godot Avatar"})


func _handle_play_audio(client: WebSocketPeer, msg_id: int, data: Dictionary) -> void:
	var audio_path = data.get("path", "")
	var visemes = data.get("visemes", null)

	if audio_path.is_empty():
		_send_error(client, msg_id, "Missing audio path")
		return

	if pirate_controller and pirate_controller.has_method("play_audio_with_visemes"):
		pirate_controller.play_audio_with_visemes(audio_path, visemes)
		_send_response(client, msg_id, {"status": "ok"})
	else:
		_send_error(client, msg_id, "Pirate controller not available")


func _handle_stop_audio(client: WebSocketPeer, msg_id: int) -> void:
	if pirate_controller and pirate_controller.has_method("stop_audio"):
		pirate_controller.stop_audio()
	_send_response(client, msg_id, {"status": "ok"})


func _handle_set_expression(client: WebSocketPeer, msg_id: int, data: Dictionary) -> void:
	var expression = data.get("expression", "neutral")

	if pirate_controller and pirate_controller.has_method("set_expression"):
		pirate_controller.set_expression(expression)
	_send_response(client, msg_id, {"status": "ok"})


func _handle_play_animation(client: WebSocketPeer, msg_id: int, data: Dictionary) -> void:
	var animation = data.get("animation", "idle")
	var loop = data.get("loop", false)

	if pirate_controller and pirate_controller.has_method("play_animation"):
		pirate_controller.play_animation(animation, loop)
	_send_response(client, msg_id, {"status": "ok"})


func _handle_set_gaze(client: WebSocketPeer, msg_id: int, data: Dictionary) -> void:
	var x = data.get("x", 0.5)
	var y = data.get("y", 0.5)
	var z = data.get("z", 1.0)

	if pirate_controller and pirate_controller.has_method("set_gaze"):
		pirate_controller.set_gaze(Vector3(x, y, z))
	_send_response(client, msg_id, {"status": "ok"})


func _handle_reset(client: WebSocketPeer, msg_id: int) -> void:
	if pirate_controller and pirate_controller.has_method("reset_to_default"):
		pirate_controller.reset_to_default()
	_send_response(client, msg_id, {"status": "ok"})


func _send_response(client: WebSocketPeer, msg_id: int, data: Dictionary) -> void:
	"""Send a successful response."""
	var response = {"id": msg_id, "status": "ok"}
	response.merge(data)
	client.send_text(JSON.stringify(response))


func _send_error(client: WebSocketPeer, msg_id: int, error_msg: String) -> void:
	"""Send an error response."""
	var response = {
		"id": msg_id,
		"status": "error",
		"error": error_msg
	}
	client.send_text(JSON.stringify(response))


func broadcast(message: Dictionary) -> void:
	"""Broadcast a message to all connected clients."""
	var json_str = JSON.stringify(message)
	for client in _clients:
		if client.get_ready_state() == WebSocketPeer.STATE_OPEN:
			client.send_text(json_str)
