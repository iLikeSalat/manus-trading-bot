"""
Module de parsing des signaux de trading Telegram.

Ce module contient les classes et fonctions nécessaires pour extraire
les informations structurées des signaux de trading reçus via Telegram.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TradingSignal:
    """Représente un signal de trading structuré."""
    
    symbol: str
    direction: str  # "LONG" ou "SHORT"
    entry_min: float
    entry_max: float
    take_profit_levels: List[float]
    stop_loss: float
    raw_text: str
    
    def __str__(self) -> str:
        """Retourne une représentation en chaîne du signal."""
        tp_str = ", ".join([str(tp) for tp in self.take_profit_levels])
        return (
            f"{self.symbol} {self.direction}\n"
            f"Entry: {self.entry_min} - {self.entry_max}\n"
            f"Take Profits: {tp_str}\n"
            f"Stop Loss: {self.stop_loss}"
        )


class SignalParser:
    """
    Classe pour analyser et extraire les informations des signaux de trading.
    
    Cette classe utilise des expressions régulières pour extraire les informations
    structurées des messages de signaux de trading.
    """
    
    def __init__(self):
        """Initialise le parser avec les patterns regex nécessaires."""
        # Définir les patterns regex pour l'extraction
        self.direction_pattern = re.compile(r'(LONG|SHORT)', re.IGNORECASE)
        self.symbol_pattern = re.compile(r'🟢\s+(\w+)\s+', re.IGNORECASE)
        self.entry_pattern = re.compile(r'Entry price:\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)')
        self.tp_pattern = re.compile(r'TP:(.*?)(?:🛑|$)', re.DOTALL)
        self.sl_pattern = re.compile(r'SL\s+(\d+(?:\.\d+)?)')
    
    def parse(self, message_text: str) -> Optional[TradingSignal]:
        """
        Parse un message texte et extrait les informations du signal de trading.
        
        Args:
            message_text: Le texte du message à analyser.
            
        Returns:
            Un objet TradingSignal contenant les informations extraites,
            ou None si le parsing a échoué.
        """
        try:
            # Extraire la direction
            direction_match = self.direction_pattern.search(message_text)
            if not direction_match:
                print("Direction (LONG/SHORT) non trouvée dans le message")
                return None
            direction = direction_match.group(1).upper()
            
            # Extraire le symbole
            symbol_match = self.symbol_pattern.search(message_text)
            if not symbol_match:
                print("Symbole non trouvé dans le message")
                return None
            symbol = symbol_match.group(1)
            
            # Extraire la plage de prix d'entrée
            entry_match = self.entry_pattern.search(message_text)
            if not entry_match:
                print("Plage de prix d'entrée non trouvée dans le message")
                return None
            entry_min = float(entry_match.group(1))
            entry_max = float(entry_match.group(2))
            
            # Extraire les niveaux de Take Profit
            tp_levels = []
            tp_section = self.tp_pattern.search(message_text)
            if tp_section:
                tp_text = tp_section.group(1)
                tp_levels = [float(level.strip()) for level in re.findall(r'(\d+(?:\.\d+)?)', tp_text)]
            
            if not tp_levels:
                print("Niveaux de Take Profit non trouvés dans le message")
                return None
            
            # Extraire le Stop Loss
            sl_match = self.sl_pattern.search(message_text)
            if not sl_match:
                print("Stop Loss non trouvé dans le message")
                return None
            stop_loss = float(sl_match.group(1))
            
            # Créer et retourner l'objet signal
            return TradingSignal(
                symbol=symbol,
                direction=direction,
                entry_min=entry_min,
                entry_max=entry_max,
                take_profit_levels=tp_levels,
                stop_loss=stop_loss,
                raw_text=message_text
            )
        except Exception as e:
            print(f"Erreur lors du parsing du signal: {str(e)}")
            return None


class SignalValidator:
    """
    Classe pour valider les signaux de trading extraits.
    
    Cette classe vérifie la validité et la cohérence des signaux
    de trading extraits par le SignalParser.
    """
    
    def validate(self, signal: TradingSignal) -> List[str]:
        """
        Valide un signal de trading.
        
        Args:
            signal: Le signal de trading à valider.
            
        Returns:
            Une liste d'erreurs trouvées. Liste vide si le signal est valide.
        """
        errors = []
        
        # Vérifier la présence des champs obligatoires
        if not signal.symbol:
            errors.append("Symbole manquant")
        
        if not signal.direction:
            errors.append("Direction (LONG/SHORT) manquante")
        
        if not signal.entry_min or not signal.entry_max:
            errors.append("Plage de prix d'entrée incomplète")
        
        if not signal.take_profit_levels or len(signal.take_profit_levels) == 0:
            errors.append("Niveaux de Take Profit manquants")
        
        if not signal.stop_loss:
            errors.append("Stop Loss manquant")
        
        # Vérifier la cohérence des prix
        if signal.direction == "LONG":
            if signal.stop_loss >= signal.entry_min:
                errors.append("Stop Loss doit être inférieur au prix d'entrée pour un LONG")
            
            for tp in signal.take_profit_levels:
                if tp <= signal.entry_max:
                    errors.append(f"Take Profit {tp} doit être supérieur au prix d'entrée pour un LONG")
        else:  # SHORT
            if signal.stop_loss <= signal.entry_max:
                errors.append("Stop Loss doit être supérieur au prix d'entrée pour un SHORT")
            
            for tp in signal.take_profit_levels:
                if tp >= signal.entry_min:
                    errors.append(f"Take Profit {tp} doit être inférieur au prix d'entrée pour un SHORT")
        
        return errors


def main():
    """Fonction principale pour tester le module."""
    # Exemple de signal
    example_signal = """
    🟢 ETH LONG 

    🎯Entry price: 2255 - 2373

    TP:
    2500  
    2601  
    2770  
    3000  
    3180  
    3300  

    🛑 SL 2150
    """
    
    # Créer le parser et parser le signal
    parser = SignalParser()
    signal = parser.parse(example_signal)
    
    if signal:
        print("Signal parsé avec succès:")
        print(signal)
        
        # Valider le signal
        validator = SignalValidator()
        errors = validator.validate(signal)
        
        if errors:
            print("\nErreurs de validation:")
            for error in errors:
                print(f"- {error}")
        else:
            print("\nLe signal est valide.")
    else:
        print("Échec du parsing du signal.")


if __name__ == "__main__":
    main()
