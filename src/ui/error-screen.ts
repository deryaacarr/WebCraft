/** Replaces the page with a full-screen, human-readable error message. */
export function showErrorScreen(title: string, details: string): void {
  const root = document.createElement('div');
  root.setAttribute('role', 'alert');
  Object.assign(root.style, {
    position: 'fixed',
    inset: '0',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '16px',
    padding: '24px',
    boxSizing: 'border-box',
    background: '#111',
    color: '#eee',
    font: '16px/1.5 system-ui, sans-serif',
    textAlign: 'center',
    zIndex: '1000',
  } satisfies Partial<CSSStyleDeclaration>);

  const heading = document.createElement('h1');
  heading.textContent = title;
  heading.style.margin = '0';
  heading.style.fontSize = '28px';

  const body = document.createElement('p');
  body.textContent = details;
  body.style.maxWidth = '640px';
  body.style.margin = '0';
  body.style.whiteSpace = 'pre-line';
  body.style.color = '#bbb';

  root.append(heading, body);
  document.body.replaceChildren(root);
}
