export function shuffle<T>(a: T[]): T[] {
    for (let i = a.length; i; i--) {
        let j = Math.floor(Math.random() * i);
        [a[i - 1], a[j]] = [a[j], a[i - 1]];
    }
    return a;
}

export function squareArray<T>(w: number, h: number, v: () => T): T[][] {
    return new Array(h).fill(new Array(w).fill(v()));
}

export function turn(l: number, i: number): number[] {
    let a = new Array(l).fill(0);
    a[i] = 1;
    return a;
}

export const roundResult = (a: number[]): number[] =>
    a.length === 1 ?
        [Math.round(a[0])] :
        turn(a.length, a.indexOf(Math.max(...a)));