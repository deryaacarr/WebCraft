import { WorkerHandler, type WorkerRequest } from './protocol';

// Typed view of the dedicated worker scope (the DOM lib is active project-wide).
const scope = self as unknown as {
  onmessage: ((e: MessageEvent<WorkerRequest>) => void) | null;
  postMessage(message: unknown, transfer: Transferable[]): void;
};

const handler = new WorkerHandler();

scope.onmessage = (e) => {
  const reply = handler.handle(e.data);
  // Transfer list: the chunk's ArrayBuffer moves to the main thread without a copy.
  if (reply) scope.postMessage(reply.response, reply.transfer);
};
